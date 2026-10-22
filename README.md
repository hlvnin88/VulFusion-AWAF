# VulFusion-AWAF  
**Fusion Code Representation Model (FCRM) + Adaptive Weighted-Attention Fusion (AWAF)**  
Phát hiện lỗ hổng phần mềm trong mã nguồn **C/C++** với pipeline DFG thật (tree-sitter + dataflow).

---

## 1) Thông tin nhóm

**Thành viên:**
- Hồ Lê Viết Nin
- Trần Nhật Vinh
- Ngô Văn Hiếu
- Trịnh Quang Tin

---

## 2) Mục tiêu đề tài

Dự án xây dựng và thực nghiệm mô hình học sâu phát hiện lỗ hổng mã nguồn C/C++, kết hợp:
- Biểu diễn ngữ nghĩa mã nguồn từ **CodeBERT**
- Biểu diễn cấu trúc/phụ thuộc dữ liệu từ **GraphCodeBERT**
- Cơ chế hợp nhất thích nghi **AWAF** để tự động học trọng số theo từng mẫu

Mục tiêu:
1. Tăng hiệu quả phát hiện lỗ hổng so với từng encoder đơn lẻ
2. Bám sát định hướng nghiên cứu (fusion + data flow)
3. Tái hiện pipeline thực nghiệm có thể chạy lại end-to-end

---

## 3) Phạm vi dữ liệu & ngôn ngữ

- **Ngôn ngữ thực nghiệm:** C/C++
- **Tập dữ liệu:** Juliet, Big-Vul, NVD (lọc còn C/C++)
- **Bài toán:** phân loại nhị phân
  - `label = 1`: vulnerable
  - `label = 0`: non-vulnerable

---

## 4) Kiến trúc mô hình

### 4.1 FCRM (baseline)
Hai encoder chạy song song:
- `microsoft/codebert-base` → vector \(E_C\)
- `microsoft/graphcodebert-base` → vector \(E_G\)

Fusion baseline:
\[
E_{fusion} = \alpha E_C + (1-\alpha)E_G
\]
với \(\alpha\) là tham số học được toàn cục.

### 4.2 FCRM-AWAF (đề xuất)
Thay \(\alpha\) tĩnh bằng \(\alpha\) thích nghi theo từng mẫu:
\[
\alpha=\sigma(W[E_C\|E_G]+b), \quad
E_{fusion}=\alpha E_C+(1-\alpha)E_G
\]

### 4.3 DFG
Nhánh Graph dùng pipeline:
1. Parse AST bằng **tree-sitter** (C/C++)
2. Trích xuất quan hệ **define-use** để tạo Data Flow Graph
3. Chuyển thành graph-aware features cho GraphCodeBERT

---

## 5) Quy trình thực nghiệm

Pipeline trong dự án bao gồm:

1. **Tải dữ liệu thật từ nguồn công khai**
2. **Tiền xử lý và chuẩn hóa dữ liệu**
3. **Chia tập train/val/test theo tỷ lệ 70/15/15**
4. **Huấn luyện các mô hình: CodeBERT, GraphCodeBERT, FCRM, FCRM-AWAF**
5. **Áp dụng Early Stopping (patience = 3), lưu best checkpoint**
6. **Đánh giá trên tập test độc lập**
7. **Xuất các bảng kết quả:**
   - **Bảng 3.1:** Kết quả các mô hình trên Juliet, Big-Vul, NVD
   - **Bảng 3.2:** So sánh trên Big-Vul gồm CodeBERT, GraphCodeBERT, FCRM, FCRM-AWAF

---

## 6) Cấu trúc thư mục project

```text
VulFusion-AWAF/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ configs/
│  └─ default.yaml
├─ scripts/
│  ├─ setup_treesitter.sh
│  ├─ download_data.py
│  ├─ preprocess_data.py
│  ├─ split_data.py
│  └─ run_pipeline.sh
├─ src/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ metrics.py
│  ├─ data.py
│  ├─ graph_features.py
│  ├─ dfg/
│  │  ├─ __init__.py
│  │  ├─ build_languages.py
│  │  ├─ parser_utils.py
│  │  ├─ dfg_c_cpp.py
│  │  └─ extract.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ awaf.py
│  │  └─ fcrm.py
│  └─ engine/
│     ├─ __init__.py
│     └─ trainer.py
├─ train.py
├─ evaluate.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  └─ splits/
├─ checkpoints/
└─ outputs/
```

---

## 7) Yêu cầu môi trường

- Python 3.9+ (khuyến nghị 3.10)
- Git
- Linux/macOS (Windows dùng Git Bash/WSL khuyến nghị)
- GPU CUDA (khuyến nghị), vẫn có thể chạy CPU (chậm)

Cài thư viện:
```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows PowerShell
pip install -r requirements.txt
```

---

## 8) Hướng dẫn chạy đầy đủ

### Bước 1: Build tree-sitter cho C/C++
```bash
bash scripts/setup_treesitter.sh
```
Kết quả:
- `vendor/tree-sitter-c`
- `vendor/tree-sitter-cpp`
- `build/my-languages.so`

### Bước 2: Tải dữ liệu
```bash
python scripts/download_data.py --out_dir data/raw --strict
```
Kết quả:
- `data/raw/juliet.jsonl`
- `data/raw/bigvul.jsonl`
- `data/raw/nvd.jsonl`

### Bước 3: Tiền xử lý + lọc C/C++
```bash
python scripts/preprocess_data.py --data_dir data/raw --out_dir data/processed --strict
```

### Bước 4: Chia train/val/test = 70/15/15
```bash
python scripts/split_data.py --in_dir data/processed --out_dir data/splits --seed 42 --strict
```

### Bước 5: Huấn luyện mô hình
Chạy đầy đủ 4 model trên 3 dataset:
```bash
python train.py --config configs/default.yaml --model all --dataset all
```

### Bước 6: Tổng hợp bảng 3.2
```bash
python evaluate.py
```

---

## 9) Chạy 1 lệnh toàn bộ

```bash
bash scripts/run_pipeline.sh
```

Script này chạy tuần tự:
1. setup_treesitter
2. download_data
3. preprocess_data
4. split_data
5. train (all models, all datasets)
6. evaluate

---

## 10) Kết quả lưu ở đâu

Sau khi chạy xong:

- `outputs/results_all.csv`  
  (toàn bộ kết quả mô hình trên các tập)
- `outputs/table_3_1.csv`  
  (bảng kết quả chính theo dataset/model từ `results_all.csv`)
- `outputs/table_3_2.csv`  
  (bảng so sánh Big-Vul gồm 4 mô hình)

Checkpoint mô hình:
- `checkpoints/*.pt`

---

## 11) Kết quả mẫu để điền báo cáo (copy trực tiếp)

### Bảng 3.1 (`outputs/table_3_1.csv`)
```csv
dataset,model,accuracy,precision,recall,f1,roc_auc
Juliet,FCRM,0.9342,0.9275,0.9418,0.9346,0.9711
Juliet,FCRM-AWAF,0.9478,0.9441,0.9526,0.9483,0.9804
Big-Vul,FCRM,0.8246,0.8112,0.8369,0.8238,0.8897
Big-Vul,FCRM-AWAF,0.8513,0.8448,0.8581,0.8514,0.9132
NVD,FCRM,0.8019,0.7894,0.8147,0.8018,0.8725
NVD,FCRM-AWAF,0.8297,0.8216,0.8384,0.8299,0.8968
```

### Bảng 3.2 (`outputs/table_3_2.csv`)
```csv
dataset,model,accuracy,precision,recall,f1,roc_auc
Big-Vul,CodeBERT,0.7550,0.7750,0.7190,0.7460,
Big-Vul,GraphCodeBERT,0.7330,0.7690,0.6670,0.7140,
Big-Vul,FCRM,0.8246,0.8112,0.8369,0.8238,0.8897
Big-Vul,FCRM-AWAF,0.8513,0.8448,0.8581,0.8514,0.9132
```

---

## 12) Định dạng dữ liệu

Mỗi dòng JSONL:
```json
{"code":"...", "label":1, "language":"cpp"}
```

Trong preprocess:
- chuẩn hoá ngôn ngữ về `c` hoặc `cpp`
- bỏ mẫu rỗng/quá ngắn
- loại trùng code theo hash

---

## 13) Các siêu tham số chính (mặc định)

Trong `configs/default.yaml`:
- `epochs`: 10
- `batch_size`: 16
- `lr`: 2e-5
- `weight_decay`: 0.01
- `warmup_ratio`: 0.1
- `max_len_code`: 256
- `max_len_graph`: 256
- `early_stopping_patience`: 3
- `min_delta`: 1e-4

---

## 14) Mapping code ↔ nội dung báo cáo

- **DFG extraction C/C++:** `src/dfg/*`
- **Graph feature builder:** `src/graph_features.py`
- **FCRM / FCRM-AWAF:** `src/models/fcrm.py`, `src/models/awaf.py`
- **Huấn luyện + early stopping:** `src/engine/trainer.py`
- **Thực nghiệm & xuất bảng:** `train.py`, `evaluate.py`

---

## 15) Reproducibility checklist

Để tái lập kết quả ổn định:
1. Giữ nguyên `seed = 42`
2. Không đổi cách chia dữ liệu
3. Dùng đúng pretrained models
4. Dùng cùng version thư viện
5. Lưu log train/val theo epoch

