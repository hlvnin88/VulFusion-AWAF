"""
Sinh 3 bảng đúng cấu trúc báo cáo:

  table_3_1.csv    ↔  TABLE III  (in-domain, FCRM vs FCRM-AWAF trên Juliet/Big-Vul/NVD)
  table_3_2.csv    ↔  TABLE IV   (Big-Vul; CodeBERT/GraphCodeBERT trích từ [23],
                                  FCRM/FCRM-AWAF từ thực nghiệm)
  table_5_cross.csv ↔ TABLE V    (cross-dataset FCRM vs FCRM-AWAF)

Lưu ý: CodeBERT/GraphCodeBERT trong TABLE IV là số trích dẫn từ Qi et al. [23]
vì Mục V-B của báo cáo nêu rõ như vậy. KHÔNG trộn với kết quả tự train
(khác tiền xử lý, khác data augmentation) — tránh so sánh sai.
"""
import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, load_yaml, ensure_dir, load_jsonl
from src.data import VulnDataset
from src.models.fcrm import FCRMBase, FCRMAWAF
from src.engine.trainer import evaluate


# Số liệu trích dẫn [23] (Qi et al.) — dùng cho TABLE IV
PUBLISHED_BIGVUL = [
    {"dataset": "Big-Vul", "model": "CodeBERT",     "accuracy": 0.755,
     "precision": 0.775, "recall": 0.719, "f1": 0.746, "roc_auc": None,
     "mcc": None, "source": "[23]"},
    {"dataset": "Big-Vul", "model": "GraphCodeBERT", "accuracy": 0.733,
     "precision": 0.769, "recall": 0.667, "f1": 0.714, "roc_auc": None,
     "mcc": None, "source": "[23]"},
]


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
        raise FileNotFoundError("Chạy train.py trước: outputs/results_all.csv chưa có.")

    df = pd.read_csv(results_csv)

    # ---- Table III: in-domain FCRM vs FCRM-AWAF ----
    t3 = df[df["model"].isin(["FCRM", "FCRM-AWAF"])].copy()
    order_ds = {"Juliet": 0, "Big-Vul": 1, "NVD": 2}
    order_md = {"FCRM": 0, "FCRM-AWAF": 1}
    t3["_od"] = t3["dataset"].map(order_ds).fillna(99)
    t3["_om"] = t3["model"].map(order_md).fillna(99)
    t3 = t3.sort_values(["_od", "_om"]).drop(columns=["_od", "_om"])
    cols_iii = ["dataset", "model", "accuracy", "precision", "recall",
                "f1", "roc_auc", "mcc"]
    cols_iii = [c for c in cols_iii if c in t3.columns]
    t3[cols_iii].to_csv(os.path.join(cfg["paths"]["output_dir"], "table_3_1.csv"),
                        index=False)
    print("[OK] saved outputs/table_3_1.csv   (TABLE III)")

    # ---- Table IV: Big-Vul, 4 model (2 published + 2 trained) ----
    big_trained = df[(df["dataset"] == "Big-Vul") &
                     (df["model"].isin(["FCRM", "FCRM-AWAF"]))].copy()
    big_trained["source"] = "This work"
    trained_rows = big_trained.to_dict(orient="records")

    t4 = pd.DataFrame(PUBLISHED_BIGVUL + trained_rows)
    order = ["CodeBERT", "GraphCodeBERT", "FCRM", "FCRM-AWAF"]
    t4["_o"] = t4["model"].apply(lambda x: order.index(x) if x in order else 99)
    t4 = t4.sort_values("_o").drop(columns="_o").reset_index(drop=True)

    missing = set(order) - set(t4["model"])
    if missing:
        print(f"[WARN] TABLE IV thiếu: {sorted(missing)} (chưa train xong?)")

    cols_iv = ["dataset", "model", "accuracy", "precision", "recall",
               "f1", "roc_auc", "source"]
    cols_iv = [c for c in cols_iv if c in t4.columns]
    t4[cols_iv].to_csv(os.path.join(cfg["paths"]["output_dir"], "table_3_2.csv"),
                       index=False)
    print("[OK] saved outputs/table_3_2.csv   (TABLE IV)")

    # ---- Table V: cross-dataset ----
    if args.skip_cross:
        return

    datasets = [("juliet", "Juliet"), ("bigvul", "Big-Vul"), ("nvd", "NVD")]
    models = [("fcrm", "FCRM"), ("fcrm_awaf", "FCRM-AWAF")]
    cross_rows = []

    for tr_dk, tr_dn in datasets:
        for te_dk, te_dn in datasets:
            if tr_dk == te_dk:
                continue
            loader = None
            for mk, mn in models:
                ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], f"{tr_dk}_{mk}.pt")
                if not os.path.exists(ckpt):
                    print(f"[SKIP] {ckpt} not found")
                    continue
                if loader is None:
                    loader = get_test_loader(cfg["paths"]["split_data_dir"], te_dk, cfg)

                model = build_model(mk, cfg).to(device)
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model.eval()
                _, m = evaluate(model, loader, device)
                cross_rows.append({
                    "train": tr_dn, "test": te_dn, "model": mn,
                    "accuracy": m["accuracy"], "precision": m["precision"],
                    "recall": m["recall"], "f1": m["f1"],
                    "roc_auc": m["roc_auc"], "mcc": m["mcc"],
                })
                print(f"[CROSS] {tr_dn:8s} -> {te_dn:8s} | {mn:10s} | F1={m['f1']:.4f}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    if cross_rows:
        pd.DataFrame(cross_rows).to_csv(
            os.path.join(cfg["paths"]["output_dir"], "table_5_cross.csv"),
            index=False,
        )
        print("[OK] saved outputs/table_5_cross.csv   (TABLE V)")
    else:
        print("[WARN] Không có cross-dataset nào — kiểm tra checkpoint.")


if __name__ == "__main__":
    main()