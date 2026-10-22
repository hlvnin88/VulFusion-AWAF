import os
import argparse
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel

from src.utils import set_seed, load_yaml, ensure_dir, load_jsonl
from src.data import VulnDataset
from src.models.fcrm import FCRMBase, FCRMAWAF
from src.engine.trainer import train_engine, evaluate


class SingleEncoderClassifier(nn.Module):
    """
    Baseline for CodeBERT:
    encoder -> CLS -> MLP(768->512->256->1)
    """
    def __init__(self, model_name: str, hidden_dim=768, dropout=0.2, mlp_hidden_dims=(512, 256)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        layers = []
        cur = hidden_dim
        for h in mlp_hidden_dims:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]
            cur = h
        layers += [nn.Linear(cur, 1)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids_c, attention_mask_c, input_ids_g=None, attention_mask_g=None, position_idx_g=None):
        out = self.encoder(input_ids=input_ids_c, attention_mask=attention_mask_c)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).squeeze(-1)


class GraphSingleEncoderClassifier(nn.Module):
    """
    GraphCodeBERT baseline:
    use graph branch tensors (input_ids_g, attention_mask_g)
    """
    def __init__(self, model_name: str, hidden_dim=768, dropout=0.2, mlp_hidden_dims=(512, 256)):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        layers = []
        cur = hidden_dim
        for h in mlp_hidden_dims:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]
            cur = h
        layers += [nn.Linear(cur, 1)]
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids_c=None, attention_mask_c=None, input_ids_g=None, attention_mask_g=None, position_idx_g=None):
        out = self.encoder(input_ids=input_ids_g, attention_mask=attention_mask_g)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls).squeeze(-1)


def build_model(model_key, cfg):
    m = cfg["models"]
    hidden_dim = m["hidden_dim"]
    dropout = m["dropout"]
    mlp_hidden = tuple(m["mlp_hidden_dims"])

    if model_key == "fcrm":
        return FCRMBase(
            m["codebert_name"],
            m["graphcodebert_name"],
            hidden_dim,
            dropout,
            mlp_hidden,
        ), "FCRM"

    if model_key == "fcrm_awaf":
        return FCRMAWAF(
            m["codebert_name"],
            m["graphcodebert_name"],
            hidden_dim,
            dropout,
            mlp_hidden,
        ), "FCRM-AWAF"

    if model_key == "codebert":
        return SingleEncoderClassifier(
            m["codebert_name"],
            hidden_dim,
            dropout,
            mlp_hidden,
        ), "CodeBERT"

    if model_key == "graphcodebert":
        return GraphSingleEncoderClassifier(
            m["graphcodebert_name"],
            hidden_dim,
            dropout,
            mlp_hidden,
        ), "GraphCodeBERT"

    raise ValueError(f"Unknown model key: {model_key}")


def get_loaders(split_root, ds_key, cfg):
    tr = load_jsonl(os.path.join(split_root, ds_key, "train.jsonl"))
    va = load_jsonl(os.path.join(split_root, ds_key, "val.jsonl"))
    te = load_jsonl(os.path.join(split_root, ds_key, "test.jsonl"))

    m, t, p = cfg["models"], cfg["train"], cfg["paths"]
    tr_ds = VulnDataset(tr, m["codebert_name"], m["graphcodebert_name"], p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])
    va_ds = VulnDataset(va, m["codebert_name"], m["graphcodebert_name"], p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])
    te_ds = VulnDataset(te, m["codebert_name"], m["graphcodebert_name"], p["tree_sitter_lib"], t["max_len_code"], t["max_len_graph"])

    return (
        DataLoader(tr_ds, batch_size=t["batch_size"], shuffle=True, num_workers=2),
        DataLoader(va_ds, batch_size=t["batch_size"], shuffle=False, num_workers=2),
        DataLoader(te_ds, batch_size=t["batch_size"], shuffle=False, num_workers=2),
    )


def append_results(csv_path, row):
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        df_old = df_old[~((df_old["dataset"] == row["dataset"]) & (df_old["model"] == row["model"]))]
        df_new = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])

    df_new.to_csv(csv_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument(
        "--model",
        default="all",
        choices=["all", "codebert", "graphcodebert", "fcrm", "fcrm_awaf"],
        help="Model to train/eval. 'all' runs 4 models.",
    )
    ap.add_argument(
        "--dataset",
        default="all",
        choices=["all", "juliet", "bigvul", "nvd"],
        help="Dataset to run. 'all' runs 3 datasets.",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])

    device = cfg["device"] if torch.cuda.is_available() else "cpu"
    ensure_dir(cfg["paths"]["checkpoint_dir"])
    ensure_dir(cfg["paths"]["output_dir"])

    all_datasets = [("juliet", "Juliet"), ("bigvul", "Big-Vul"), ("nvd", "NVD")]
    all_models = [
        ("codebert", "CodeBERT"),
        ("graphcodebert", "GraphCodeBERT"),
        ("fcrm", "FCRM"),
        ("fcrm_awaf", "FCRM-AWAF"),
    ]

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
            test_loss, test_m = evaluate(model, te_ld, device)

            row = {
                "dataset": dn,
                "model": model_name,
                "accuracy": test_m["accuracy"],
                "precision": test_m["precision"],
                "recall": test_m["recall"],
                "f1": test_m["f1"],
                "roc_auc": test_m["roc_auc"],
            }
            append_results(results_csv, row)

            print(
                f"[DONE] {dn} | {model_name} | "
                f"Acc={row['accuracy']:.4f} Prec={row['precision']:.4f} "
                f"Rec={row['recall']:.4f} F1={row['f1']:.4f} AUC={row['roc_auc']:.4f}"
            )

    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        df.to_csv(os.path.join(cfg["paths"]["output_dir"], "table_3_1.csv"), index=False)
        print(f"\n[OK] saved {results_csv}")
        print(f"[OK] saved {os.path.join(cfg['paths']['output_dir'], 'table_3_1.csv')}")


if __name__ == "__main__":
    main()