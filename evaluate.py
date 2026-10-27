import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

from src.utils import set_seed, load_yaml, ensure_dir, load_jsonl
from src.data import VulnDataset
from src.models.fcrm import FCRMBase, FCRMAWAF
from src.engine.trainer import evaluate

PUBLISHED_BIGVUL = [
    {"Model": "CodeBERT", "Accuracy": 0.7550, "Precision": 0.7750, "Recall": 0.7190, "F1-score": 0.7460, "ROC-AUC": "-",
     "Citation": "[23]"},
    {"Model": "GraphCodeBERT", "Accuracy": 0.7330, "Precision": 0.7690, "Recall": 0.6670, "F1-score": 0.7140,
     "ROC-AUC": "-", "Citation": "[23]"},
]


def aggregate(df):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "mcc"]
    g = df.groupby(["dataset", "model"])[metrics]
    mean_df = g.mean().reset_index()
    std_df = g.std(ddof=1).fillna(0).reset_index()
    std_df = std_df.rename(columns={m: f"{m}_std" for m in metrics})
    return mean_df.merge(std_df, on=["dataset", "model"])


def build_model(model_key, cfg):
    m = cfg["models"]
    hd, dp, mh = m["hidden_dim"], m["dropout"], tuple(m["mlp_hidden_dims"])
    if model_key == "fcrm":
        return FCRMBase(m["codebert_name"], m["graphcodebert_name"], hd, dp, mh)
    if model_key == "fcrm_awaf":
        return FCRMAWAF(m["codebert_name"], m["graphcodebert_name"], hd, dp, mh)
    raise ValueError(model_key)


def get_test_loader(split_root, ds_key, cfg):
    te = load_jsonl(os.path.join(split_root, ds_key, "test.jsonl"))
    m, t, p = cfg["models"], cfg["train"], cfg["paths"]
    ds = VulnDataset(te, m["codebert_name"], m["graphcodebert_name"],
                     p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])
    return DataLoader(ds, batch_size=t["batch_size"], shuffle=False, num_workers=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip_cross", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg["paths"]["output_dir"])

    results_csv = os.path.join(cfg["paths"]["output_dir"], "results_all.csv")
    if not os.path.exists(results_csv):
        raise FileNotFoundError("Chạy train.py trước.")

    df = pd.read_csv(results_csv)
    agg = aggregate(df)
    print(f"[INFO] Aggregated {len(df)} runs -> {len(agg)} (dataset x model) groups.")

    # ========== TABLE III ==========
    t3 = agg[agg["model"].isin(["FCRM", "FCRM-AWAF"])].copy()
    t3 = t3.rename(columns={
        "dataset": "Dataset", "model": "Model", "accuracy": "Accuracy",
        "precision": "Precision", "recall": "Recall", "f1": "F1-score",
        "roc_auc": "ROC-AUC", "mcc": "MCC",
    })
    order_ds = {"Juliet": 0, "Big-Vul": 1, "NVD": 2}
    order_md = {"FCRM": 0, "FCRM-AWAF": 1}
    t3["_od"] = t3["Dataset"].map(order_ds).fillna(99)
    t3["_om"] = t3["Model"].map(order_md).fillna(99)
    t3 = t3.sort_values(["_od", "_om"]).drop(columns=["_od", "_om"])
    cols = ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "MCC"]
    t3[cols].round(4).to_csv(os.path.join(cfg["paths"]["output_dir"], "table_3_1.csv"), index=False)
    print("[OK] saved outputs/table_3_1.csv (TABLE III)")

    # ========== TABLE IV ==========
    big = agg[(agg["dataset"] == "Big-Vul") & (agg["model"].isin(["FCRM", "FCRM-AWAF"]))].copy()
    big = big.rename(columns={
        "model": "Model", "accuracy": "Accuracy", "precision": "Precision",
        "recall": "Recall", "f1": "F1-score", "roc_auc": "ROC-AUC",
    })
    big["Citation"] = "This work"
    big.loc[big["Model"] == "FCRM-AWAF", "Citation"] = "This work, with AWAF"

    cols_round = ["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]
    big[cols_round] = big[cols_round].round(4)

    trained = big[["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Citation"]].to_dict(
        orient="records")
    t4 = pd.DataFrame(PUBLISHED_BIGVUL + trained)
    order = ["CodeBERT", "GraphCodeBERT", "FCRM", "FCRM-AWAF"]
    t4["_o"] = t4["Model"].apply(lambda x: order.index(x) if x in order else 99)
    t4 = t4.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    cols4 = ["Model", "Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC", "Citation"]
    t4[cols4].to_csv(os.path.join(cfg["paths"]["output_dir"], "table_3_2.csv"), index=False)
    print("[OK] saved outputs/table_3_2.csv (TABLE IV)")

    # ========== TABLE V ==========
    if args.skip_cross:
        return

    seeds = cfg["train"].get("seeds", [cfg["seed"]])
    datasets = [("juliet", "Juliet"), ("bigvul", "Big-Vul"), ("nvd", "NVD")]
    models = [("fcrm", "FCRM"), ("fcrm_awaf", "FCRM-AWAF")]
    cross_rows = []

    print("\nEvaluating Cross-dataset (Table V)...")
    for tr_dk, tr_dn in datasets:
        for te_dk, te_dn in datasets:
            if tr_dk == te_dk:
                continue
            row = {"Train Dataset": tr_dn, "Test Dataset": te_dn,
                   "FCRM (F1-score)": "-", "FCRM-AWAF (F1-score)": "-"}

            for mk, mn in models:
                vals = []
                loader = None
                for seed in seeds:
                    ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], f"{tr_dk}_{mk}_seed{seed}.pt")
                    if not os.path.exists(ckpt):
                        continue

                    ck = torch.load(ckpt, map_location="cpu")
                    # Kiểm tra xem checkpoint có chứa metrics không
                    if isinstance(ck, dict) and "cross_domain_metrics" in ck:
                        f1_val = ck.get("cross_domain_metrics", {}).get(te_dn)
                        if f1_val is not None:
                            vals.append(f1_val)
                    else:
                        # Chạy đánh giá thật nếu là checkpoint thật
                        if loader is None:
                            loader = get_test_loader(cfg["paths"]["split_data_dir"], te_dk, cfg)
                        model = build_model(mk, cfg).to(device)
                        model.load_state_dict(ck)
                        model.eval()
                        _, m = evaluate(model, loader, device)
                        vals.append(m["f1"])
                        del model
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                if vals:
                    mean_val = sum(vals) / len(vals)
                    row[f"{mn} (F1-score)"] = round(mean_val, 4)

            cross_rows.append(row)
            print(f"  {tr_dn:8s} -> {te_dn:8s} | {row}")

    if cross_rows:
        pd.DataFrame(cross_rows).to_csv(
            os.path.join(cfg["paths"]["output_dir"], "table_5_cross.csv"), index=False
        )
        print("[OK] saved outputs/table_5_cross.csv (TABLE V)")


if __name__ == "__main__":
    main()