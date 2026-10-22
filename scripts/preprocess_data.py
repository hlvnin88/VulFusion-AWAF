#!/usr/bin/env python
import os
import argparse
import hashlib
from src.utils import load_jsonl, write_jsonl, ensure_dir


TARGET = {"c", "cpp", "c++"}


def norm_lang(x):
    l = str(x or "unknown").lower().strip()
    if l in {"cpp", "c++", "cc", "cxx"}:
        return "cpp"
    if l == "c":
        return "c"
    return l


def process(rows):
    out = []
    seen = set()
    for r in rows:
        code = str(r.get("code", "")).strip()
        if len(code) < 10:
            continue
        lang = norm_lang(r.get("language"))
        if lang not in {"c", "cpp"}:
            continue
        label = 1 if int(r.get("label", 0)) > 0 else 0
        h = hashlib.md5((lang + "::" + code).encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append({"code": code, "label": label, "language": lang})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    counts = {}

    for name in ["juliet", "bigvul", "nvd"]:
        p = os.path.join(args.data_dir, f"{name}.jsonl")
        if not os.path.exists(p):
            print(f"[WARN] missing {p}")
            counts[name] = 0
            continue
        rows = load_jsonl(p)
        rows = process(rows)
        write_jsonl(os.path.join(args.out_dir, f"{name}.jsonl"), rows)
        counts[name] = len(rows)
        print(f"[OK] {name}: {len(rows)}")

    if args.strict:
        missing = [k for k, v in counts.items() if v == 0]
        if missing:
            raise SystemExit(f"[ERROR] empty processed datasets: {missing}")


if __name__ == "__main__":
    main()