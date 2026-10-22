import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, load_yaml, ensure_dir, load_jsonl
from src.data import VulnDataset
from src.models.fcrm import FCRMBase, FCRMAWAF
from src.engine.trainer import train_engine, evaluate


def build_model(model_key, cfg):
    m = cfg["models"]
    hd, dp, mh = m["hidden_dim"], m["dropout"], tuple(m["mlp_hidden_dims"])
    if model_key == "fcrm":
        return FCRMBase(m["codebert_name"], m["graphcodebert_name"], hd, dp, mh), "FCRM"
    if model_key == "fcrm_awaf":
        return FCRMAWAF(m["codebert_name"], m["graphcodebert_name"], hd, dp, mh), "FCRM-AWAF"
    raise ValueError(model_key)


def get_loaders(split_root, ds_key, cfg):
    tr = load_jsonl(os.path.join(split_root, ds_key, "train.jsonl"))
    va = load_jsonl(os.path.join(split_root, ds_key, "val.jsonl"))
    te = load_jsonl(os.path.join(split_root, ds_key, "test.jsonl"))
    m, t, p = cfg["models"], cfg["train"], cfg["paths"]

    def _mk(rows, shuffle):
        ds = VulnDataset(rows, m["codebert_name"], m["graphcodebert_name"],
                         p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])
        # TỐI ƯU: Tăng num_workers để đa luồng (Windows khuyến nghị 2-4)
        return DataLoader(ds, batch_size=t["batch_size"], shuffle=shuffle,
                          num_workers=4, persistent_workers=True)

    return _mk(tr, True), _mk(va, False), _mk(te, False)


def append_results(csv_path, row):
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_old = df_old[~((df_old["dataset"] == row["dataset"]) &
                          (df_old["model"] == row["model"]))]
        df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])
    order = {"Juliet": 0, "Big-Vul": 1, "NVD": 2}
    df_new["_o"] = df_new["dataset"].map(order).fillna(99)
    df_new = df_new.sort_values(["_o", "model"]).drop(columns="_o")
    df_new.to_csv(csv_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="all",
                    choices=["all", "fcrm", "fcrm_awaf"])
    ap.add_argument("--dataset", default="all",
                    choices=["all", "juliet", "bigvul", "nvd"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    # --- KIỂM TRA THIẾT BỊ ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Đang sử dụng GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[WARN] Không tìm thấy GPU (CUDA). Đang chạy bằng CPU (sẽ rất chậm).")
    # -------------------------

    ensure_dir(cfg["paths"]["checkpoint_dir"])
    ensure_dir(cfg["paths"]["output_dir"])

    all_datasets = [("juliet", "Juliet"), ("bigvul", "Big-Vul"), ("nvd", "NVD")]
    all_models = [("fcrm", "FCRM"), ("fcrm_awaf", "FCRM-AWAF")]
    datasets = all_datasets if args.dataset == "all" else [x for x in all_datasets if x[0] == args.dataset]
    models = all_models if args.model == "all" else [x for x in all_models if x[0] == args.model]

    results_csv = os.path.join(cfg["paths"]["output_dir"], "results_all.csv")

    for mk, mn in models:
        for dk, dn in datasets:
            sdir = os.path.join(cfg["paths"]["split_data_dir"], dk)
            if not os.path.exists(sdir):
                print(f"[WARN] missing split dir: {sdir}")
                continue

            print(f"\n=== Running model={mn} on dataset={dn} ===")
            tr_ld, va_ld, te_ld = get_loaders(cfg["paths"]["split_data_dir"], dk, cfg)

            model, model_name = build_model(mk, cfg)
            model = model.to(device)

            ckpt = os.path.join(cfg["paths"]["checkpoint_dir"], f"{dk}_{mk}.pt")
            model = train_engine(model, tr_ld, va_ld, cfg["train"], device, ckpt)
            _, test_m = evaluate(model, te_ld, device)

            row = {
                "dataset": dn, "model": model_name,
                "accuracy": test_m["accuracy"], "precision": test_m["precision"],
                "recall": test_m["recall"], "f1": test_m["f1"],
                "roc_auc": test_m["roc_auc"], "mcc": test_m["mcc"],
            }
            append_results(results_csv, row)
            print(f"[DONE] {dn} | {model_name} | "
                  f"Acc={row['accuracy']:.4f} F1={row['f1']:.4f} "
                  f"AUC={row['roc_auc']:.4f} MCC={row['mcc']:.4f}")

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if os.path.exists(results_csv):
        print(f"\n[OK] saved {results_csv}")


if __name__ == "__main__":
    main()