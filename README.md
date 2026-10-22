# VulFusion-AWAF

**FCRM-AWAF: Adaptive Weighted-Attention Fusion of Semantic and Structural Representations for Source Code Vulnerability Detection**

## Kiến trúc & công thức

- **Semantic**: `microsoft/codebert-base` → `E_C` (Eq.1-2)
- **Structural**: `microsoft/graphcodebert-base` (DFG từ tree-sitter) → `E_G` (Eq.3-5)
- **AWAF** (Eq.7-9): `α = σ(W[E_C‖E_G]+b)` — **scalar per sample**
- **Fusion** (Eq.8): `E_fusion = α·E_C + (1-α)·E_G`
- **Classifier** (Eq.10): FC + **Softmax** (2 lớp)
- **Loss** (Eq.11): **Cross-Entropy**

## Dữ liệu
- `data/raw/{juliet,bigvul,nvd}.jsonl` — mỗi dòng `{"code","label","language"}`
- C/C++ nhị phân (0/1)

## Quy trình (Chạy trên Windows PowerShell)

1. Kích hoạt môi trường ảo và fix lỗi import module `src`:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   $env:PYTHONPATH="."
   ```

2. Build thư viện tree-sitter (Bắt buộc dùng branch `v0.23.0` để tương thích với thư viện `tree_sitter<0.22`):
   ```powershell
   New-Item -ItemType Directory -Force -Path vendor, build | Out-Null
   if (-not (Test-Path vendor\tree-sitter-c)) { git clone --branch v0.23.0 https://github.com/tree-sitter/tree-sitter-c vendor\tree-sitter-c }
   if (-not (Test-Path vendor\tree-sitter-cpp)) { git clone --branch v0.23.0 https://github.com/tree-sitter/tree-sitter-cpp vendor\tree-sitter-cpp }
   python -m src.dfg.build_languages --output build\my-languages.so --repos vendor\tree-sitter-c vendor\tree-sitter-cpp
   ```

3. Tải dữ liệu (Big-Vul, Juliet, NVD) từ HuggingFace:
   ```powershell
   python scripts\download_data.py
   ```

4. Tiền xử lý dữ liệu:
   ```powershell
   python scripts\preprocess_data.py
   ```

5. Chia tập dữ liệu (80/10/10 theo báo cáo):
   ```powershell
   python scripts\split_data.py --seed 42
   ```

6. Huấn luyện mô hình (FCRM + FCRM-AWAF, đúng Mục V-A):
   ```powershell
   python train.py --config configs\default.yaml --model all --dataset all
   ```

7. Sinh 3 bảng kết quả đúng báo cáo:
   ```powershell
   python evaluate.py --config configs\default.yaml
   ```

## Output
| File | Bảng tương ứng | Nội dung |
|---|---|---|
| `outputs/table_3_1.csv` | **TABLE III** | In-domain FCRM vs FCRM-AWAF trên 3 dataset |
| `outputs/table_3_2.csv` | **TABLE IV** | Big-Vul; CodeBERT/GraphCodeBERT trích [23], FCRM/FCRM-AWAF tự train |
| `outputs/table_5_cross.csv` | **TABLE V** | Cross-dataset (6 cặp × 2 model) |

## Hyperparameters (configs/default.yaml)
- `epochs = 20`, `batch_size = 16`, `lr = 2e-5`
- `weight_decay = 0.01`, `warmup = 0.1`
- `max_len_code = max_len_graph = 256`
- `early_stopping_patience = 3`, `min_delta = 1e-4`
- AdamW

## Khắc phục sự cố (Troubleshooting)

1. **Lỗi `ModuleNotFoundError: No module named 'src'`**: 
   - Luôn chạy `$env:PYTHONPATH="."` sau khi activate venv để các script trong thư mục `scripts/` tìm thấy thư mục `src/`.

2. **Lỗi `ValueError: Incompatible Language version 15`**:
   - Nguyên nhân do clone tree-sitter bản mới nhất (ABI 15) không tương thích với thư viện `tree_sitter 0.21.x`. 
   - Khắc phục: Xóa thư mục `vendor/tree-sitter-c` và `vendor/tree-sitter-cpp`, sau đó clone lại với cờ `--branch v0.23.0` như ở Bước 2.

3. **Tốc độ train quá chậm (35s/batch)**:
   - Nguyên nhân: Máy đang chạy bằng CPU thay vì GPU.
   - Kiểm tra bằng lệnh: `python -c "import torch; print(torch.cuda.is_available())"`
   - Nếu ra `False`, hãy cài lại PyTorch bản CUDA: `python -m pip uninstall torch -y` và `python -m pip install torch --index-url https://download.pytorch.org/whl/cu121`
   - Nếu máy không có GPU, hãy sửa `configs/default.yaml` thành `epochs: 1` và `early_stopping_patience: 1` để chạy smoke test lấy kết quả báo cáo.

## Lưu ý quan trọng
1. **Không retrain CodeBERT/GraphCodeBERT**: Mục V-B báo cáo nêu rõ số của 2 model này trích từ [23]. Nếu bạn tự train lại trên data của mình, kết quả sẽ lệch so với TABLE IV gốc.
2. **GraphCodeBERT qua HuggingFace** (`AutoModel`) thực chất là RoBERTa thuần — không có graph-attention gốc. Code truyền `position_ids` để mô phỏng. Đây là **giới hạn kỹ thuật**, không phải bug.
3. **DataLoader phải `num_workers=0`** vì DFGExtractor chứa tree-sitter Parser (C extension không pickle).
4. **Big-Vul rất mất cân bằng** (5.8% vulnerable theo báo cáo — 10,900/188,636). Nếu train trên Big-Vul gốc với BCE/CE thường, model có xu hướng đoán non-vulnerable → F1 thấp. Để đạt F1 ≈ 0.82–0.86 như báo cáo, có thể cần undersampling hoặc điều chỉnh class weight. Báo cáo không mô tả chi tiết, nên code này **không** áp dụng undersampling mặc định.
5. **Pin `tree_sitter<0.22`** trong requirements (bản 0.22+ bỏ `Language.build_library`).