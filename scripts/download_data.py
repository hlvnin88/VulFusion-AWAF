#!/usr/bin/env python
import os
import json
import argparse
from datasets import load_dataset


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def pick_code(x):
    for k in ["func", "function", "code", "source", "source_code", "text", "processed_func"]:
        if k in x and x[k] is not None and str(x[k]).strip():
            return str(x[k])
    return ""


def pick_label(x):
    for k in ["target", "label", "vul", "is_vuln", "is_vulnerable"]:
        if k in x:
            try:
                return 1 if int(x[k]) > 0 else 0
            except Exception:
                return 1 if str(x[k]).lower() in ["true", "vulnerable", "yes"] else 0
    return 0


def pick_language(x):
    for k in ["language", "lang", "programming_language"]:
        if k in x and x[k] is not None:
            return str(x[k]).lower()
    return "unknown"


def normalize(ds, limit=None):
    rows = []
    for i, x in enumerate(ds):
        if limit is not None and i >= limit:
            break
        code = pick_code(x)
        if not code:
            continue
        rows.append({"code": code, "label": pick_label(x), "language": pick_language(x)})
    return rows


def dump_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def try_load(repo, subset=None, split="train"):
    if subset:
        return load_dataset(repo, subset, split=split)
    return load_dataset(repo, split=split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    limit = None if args.limit == 0 else args.limit

    out = {"juliet": [], "bigvul": [], "nvd": []}

    try:
        out["bigvul"] = normalize(try_load("DetectVul/devign", "bigvul", "train"), limit)
    except Exception as e:
        print("[WARN] bigvul:", e)

    try:
        out["juliet"] = normalize(try_load("DetectVul/devign", "juliet", "train"), limit)
    except Exception as e:
        print("[WARN] juliet:", e)

    for repo, subset in [("DetectVul/devign", "nvd"), ("Mireu-Lab/Fan_et_al", None)]:
        if out["nvd"]:
            break
        try:
            out["nvd"] = normalize(try_load(repo, subset, "train"), limit)
        except Exception:
            pass

    for name in ["juliet", "bigvul", "nvd"]:
        rows = out[name]
        if rows:
            path = os.path.join(args.out_dir, f"{name}.jsonl")
            dump_jsonl(path, rows)
            print(f"[OK] {name}: {len(rows)}")
        else:
            print(f"[WARN] {name} missing")

    missing = [k for k, v in out.items() if len(v) == 0]
    if args.strict and missing:
        raise SystemExit(f"[ERROR] missing datasets: {missing}")


if __name__ == "__main__":
    main()