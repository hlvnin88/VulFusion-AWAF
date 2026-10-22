#!/usr/bin/env python
import os
import random
import argparse
from src.utils import load_jsonl, write_jsonl, ensure_dir


def split_rows(rows, seed=42):
    random.Random(seed).shuffle(rows)
    n = len(rows)
    n_tr = int(n * 0.7)
    n_va = int(n * 0.15)
    return rows[:n_tr], rows[n_tr:n_tr + n_va], rows[n_tr + n_va:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/processed")
    ap.add_argument("--out_dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    ok = {}
    for name in ["juliet", "bigvul", "nvd"]:
        inp = os.path.join(args.in_dir, f"{name}.jsonl")
        if not os.path.exists(inp):
            print(f"[WARN] missing {inp}")
            ok[name] = False
            continue

        rows = load_jsonl(inp)
        tr, va, te = split_rows(rows, args.seed)

        outd = os.path.join(args.out_dir, name)
        ensure_dir(outd)
        write_jsonl(os.path.join(outd, "train.jsonl"), tr)
        write_jsonl(os.path.join(outd, "val.jsonl"), va)
        write_jsonl(os.path.join(outd, "test.jsonl"), te)

        ok[name] = len(tr) > 0 and len(va) > 0 and len(te) > 0
        print(f"[OK] {name}: train={len(tr)} val={len(va)} test={len(te)}")

    if args.strict:
        missing = [k for k, v in ok.items() if not v]
        if missing:
            raise SystemExit(f"[ERROR] split missing: {missing}")


if __name__ == "__main__":
    main()