#!/usr/bin/env bash
set -e

bash scripts/setup_treesitter.sh
python scripts/download_data.py   --out_dir data/raw
python scripts/preprocess_data.py --data_dir data/raw --out_dir data/processed
python scripts/split_data.py      --in_dir data/processed --out_dir data/splits --seed 42

mkdir -p checkpoints outputs

# Chỉ train FCRM + FCRM-AWAF (đúng Mục V-A của báo cáo)
python train.py    --config configs/default.yaml --model all --dataset all
python evaluate.py --config configs/default.yaml

echo "[OK] Tables: outputs/table_3_1.csv (Table III), outputs/table_3_2.csv (Table IV), outputs/table_5_cross.csv (Table V)"