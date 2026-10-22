#!/usr/bin/env bash
set -e

bash scripts/setup_treesitter.sh
python scripts/download_data.py --out_dir data/raw --strict
python scripts/preprocess_data.py --data_dir data/raw --out_dir data/processed --strict
python scripts/split_data.py --in_dir data/processed --out_dir data/splits --seed 42 --strict

# Full real training/evaluation
python train.py --config configs/default.yaml --model all --dataset all
python evaluate.py

echo "[OK] Pipeline done. See outputs/results_all.csv, outputs/table_3_1.csv, outputs/table_3_2.csv"