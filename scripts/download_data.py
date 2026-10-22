#!/usr/bin/env python
"""
Tải trọn bộ 3 dataset: Big-Vul, Juliet, NVD từ HuggingFace.
Chạy: python scripts/download_data.py
"""
import os
import argparse
from datasets import load_dataset
from src.utils import ensure_dir, write_jsonl


def download_bigvul(out_dir):
    print("Đang tải Big-Vul (Fallback: code_x_glue)...")
    try:
        ds = load_dataset("google/code_x_glue_cc_defect_detection", split="train")
        rows = []
        for item in ds:
            code = item.get("func", "")
            if code and len(str(code).strip()) > 10:
                label = 1 if item.get("target", 0) > 0 else 0
                rows.append({"code": str(code), "label": label, "language": "c"})

        out_path = os.path.join(out_dir, "bigvul.jsonl")
        write_jsonl(out_path, rows)
        print(f"[OK] Đã lưu {len(rows)} mẫu Big-Vul vào {out_path}")
    except Exception as e:
        print(f"[LỖI] Không thể tải Big-Vul: {e}")


def download_juliet(out_dir):
    print("Đang tải Juliet (LorenzH/juliet_test_suite_c_1_3)...")
    try:
        ds = load_dataset("LorenzH/juliet_test_suite_c_1_3", split="train")
        rows = []
        for item in ds:
            bad_code = item.get("bad")
            good_code = item.get("good")
            if bad_code and len(str(bad_code).strip()) > 10:
                rows.append({"code": str(bad_code), "label": 1, "language": "c"})
            if good_code and len(str(good_code).strip()) > 10:
                rows.append({"code": str(good_code), "label": 0, "language": "c"})

        out_path = os.path.join(out_dir, "juliet.jsonl")
        write_jsonl(out_path, rows)
        print(f"[OK] Đã lưu {len(rows)} mẫu Juliet vào {out_path}")
    except Exception as e:
        print(f"[LỖI] Không thể tải Juliet: {e}")


def download_nvd(out_dir):
    print("Đang tải NVD (arag0rn/SecVulEval)...")
    try:
        ds = load_dataset("arag0rn/SecVulEval", split="train")
        rows = []
        for item in ds:
            code = item.get("func_body")
            is_vul = item.get("is_vulnerable")
            if code and len(str(code).strip()) > 10:
                label = 1 if is_vul else 0
                rows.append({"code": str(code), "label": label, "language": "c"})

        out_path = os.path.join(out_dir, "nvd.jsonl")
        write_jsonl(out_path, rows)
        print(f"[OK] Đã lưu {len(rows)} mẫu NVD vào {out_path}")
    except Exception as e:
        print(f"[LỖI] Không thể tải NVD: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/raw")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    download_bigvul(args.out_dir)
    download_juliet(args.out_dir)
    download_nvd(args.out_dir)

    print("\n[HOÀN TẤT] Dữ liệu đã sẵn sàng trong thư mục data/raw/")
    print("Bước tiếp theo:")
    print("  python scripts/preprocess_data.py")
    print("  python scripts/split_data.py --seed 42")


if __name__ == "__main__":
    main()