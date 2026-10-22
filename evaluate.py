import os
import pandas as pd

OUT = "outputs"

def main():
    p = os.path.join(OUT, "results_all.csv")
    if not os.path.exists(p):
        raise FileNotFoundError("Run train.py first: outputs/results_all.csv not found.")

    df = pd.read_csv(p)

    required_models = {"CodeBERT", "GraphCodeBERT", "FCRM", "FCRM-AWAF"}
    big = df[df["dataset"] == "Big-Vul"].copy()

    missing = required_models - set(big["model"].unique())
    if missing:
        raise ValueError(
            f"Missing Big-Vul results for models: {sorted(missing)}. "
            "Please train/eval them first so table_3_2 is fully real."
        )

    cols = ["dataset", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    t32 = big[cols].copy()

    # order rows for report
    order = ["CodeBERT", "GraphCodeBERT", "FCRM", "FCRM-AWAF"]
    t32["model"] = pd.Categorical(t32["model"], categories=order, ordered=True)
    t32 = t32.sort_values("model").reset_index(drop=True)

    out_path = os.path.join(OUT, "table_3_2.csv")
    t32.to_csv(out_path, index=False)

    print(f"[OK] saved {out_path}")
    print(t32)

if __name__ == "__main__":
    main()