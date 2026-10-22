# VulFusion-AWAF

**FCRM-AWAF: Adaptive Weighted-Attention Fusion of Semantic and Structural Representations for Source Code Vulnerability Detection**

Official implementation of the paper "FCRM-AWAF".

## 1. Abstract

VulFusion-AWAF is a hybrid representation model for source code vulnerability detection. Instead of relying on a single representation or fixed fusion strategies, this framework combines semantic representations from **CodeBERT** and structural representations from **GraphCodeBERT** through an **Adaptive Weighted-Attention Fusion (AWAF)** mechanism. 

The framework:
- Extracts semantic features (\(E_C\)) from source code using `microsoft/codebert-base`.
- Extracts structural features (\(E_G\)) using `microsoft/graphcodebert-base` and Data Flow Graphs (DFG) extracted via `tree-sitter`.
- Dynamically computes sample-specific fusion weights: \(\alpha = \sigma(W[E_C \| E_G] + b)\).
- Constructs the final representation: \(E_{fusion} = \alpha E_C + (1-\alpha)E_G\).
- Evaluates the model under in-domain and cross-dataset (Distribution Shift) settings.

## 2. Repository Structure

```text
VulFusion-AWAF/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml               # All hyperparameters (epochs, lr, etc.)
├── scripts/
│   ├── setup_treesitter.sh        # Bash script to build tree-sitter (Linux)
│   ├── download_data.py           # Step 1: Download Big-Vul, Juliet, NVD
│   ├── preprocess_data.py         # Step 2: Filter C/C++, remove duplicates
│   └── split_data.py              # Step 3: Split 80/10/10
├── src/
│   ├── utils.py, metrics.py       # Helper functions
│   ├── data.py                    # PyTorch Dataset class
│   ├── graph_features.py          # DFG to GraphCodeBERT inputs
│   ├── dfg/                       # Tree-sitter parser and DFG extractor
│   ├── models/                    # AWAF module and FCRM models
│   └── engine/                    # Training loop and evaluation
├── train.py                       # Step 4: Train FCRM & FCRM-AWAF
├── evaluate.py                    # Step 5: Generate Tables III, IV, V
└── checkpoints/                   # Saved model weights
```

## 3. Data Sources

The pipeline automatically downloads the three datasets via HuggingFace:

| Dataset | Source (HuggingFace) | Description |
| :--- | :--- | :--- |
| **Big-Vul** | `google/code_x_glue_cc_defect_detection` | Real-world C/C++ functions (Fallback source). |
| **Juliet** | `LorenzH/juliet_test_suite_c_1_3` | Synthetic test cases with clear good/bad labels. |
| **NVD** | `arag0rn/SecVulEval` | Real-world functions collected from NVD. |

*Format: `{"code": "...", "label": 0/1, "language": "c"}`*

## 4. Requirements

- **Python 3.11** (Important: Do not use 3.12+, as `tokenizers` and `torch` might lack pre-built wheels).
- **OS**: Windows (tested on PowerShell) / Linux / macOS.
- **Compiler**: Visual Studio Build Tools (C++) for building `tree-sitter`.
- **Hardware**: NVIDIA GPU (Recommended for training). CPU works but will be extremely slow.

## 5. Installation

### Step 5.1: Clone the repository
```bash
git clone https://github.com/hlvnin88/VulFusion-AWAF.git
cd VulFusion-AWAF
```

### Step 5.2: Setup Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH="."
```
**Linux/macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
export PYTHONPATH="."
```

### Step 5.3: Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
*(Note: `requirements.txt` pins `tokenizers==0.19.1` and `transformers==4.40.2` to avoid dependency resolution errors).*

### Step 5.4: Build Tree-sitter (Crucial)
You **must** use `v0.23.0` to maintain ABI compatibility with `tree_sitter<0.22`.
```powershell
New-Item -ItemType Directory -Force -Path vendor, build | Out-Null
if (-not (Test-Path vendor\tree-sitter-c)) { git clone --branch v0.23.0 https://github.com/tree-sitter/tree-sitter-c vendor\tree-sitter-c }
if (-not (Test-Path vendor\tree-sitter-cpp)) { git clone --branch v0.23.0 https://github.com/tree-sitter/tree-sitter-cpp vendor\tree-sitter-cpp }
python -m src.dfg.build_languages --output build\my-languages.so --repos vendor\tree-sitter-c vendor\tree-sitter-cpp
```

## 6. Quick Start

Run the following commands sequentially to reproduce the results:

```powershell
# Step 1: Download and format Big-Vul, Juliet, and NVD
python scripts\download_data.py

# Step 2: Preprocess (filter C/C++, remove duplicates)
python scripts\preprocess_data.py

# Step 3: Split datasets (80/10/10)
python scripts\split_data.py --seed 42

# Step 4: Train FCRM and FCRM-AWAF on all datasets
python train.py --config configs\default.yaml --model all --dataset all

# Step 5: Generate Tables III, IV, and V
python evaluate.py --config configs\default.yaml
```

## 7. Pipeline Overview

| Step | Description | Script |
| :--- | :--- | :--- |
| 1 | Download datasets (Big-Vul, Juliet, NVD) | `scripts/download_data.py` |
| 2 | Preprocess, clean, and deduplicate C/C++ code | `scripts/preprocess_data.py` |
| 3 | Split into Train/Val/Test (80/10/10) | `scripts/split_data.py` |
| 4 | Train FCRM (Baseline) and FCRM-AWAF (Proposed) | `train.py` |
| 5 | Evaluate in-domain, compare with published results, and run cross-dataset testing | `evaluate.py` |

## 8. Output Files

All results are saved in the `outputs/` directory:

| File | Paper Table | Description |
| :--- | :--- | :--- |
| `table_3_1.csv` | **TABLE III** | In-domain performance of FCRM vs FCRM-AWAF on Juliet, Big-Vul, NVD. |
| `table_3_2.csv` | **TABLE IV** | Comparison on Big-Vul (includes CodeBERT/GraphCodeBERT from [23]). |
| `table_5_cross.csv` | **TABLE V** | Cross-dataset generalization (e.g., Train on Juliet \(\rightarrow\) Test on Big-Vul). |

Model weights are saved in `checkpoints/`.

## 9. Environment & Hyperparameters

Configured in `configs/default.yaml`:
- `epochs`: 20, `batch_size`: 16, `lr`: 2e-5
- `weight_decay`: 0.01, `warmup_ratio`: 0.1
- `max_len_code`: 256, `max_len_graph`: 256
- `early_stopping_patience`: 3, `min_delta`: 1e-4
- Optimizer: AdamW

## 10. Troubleshooting

1. **Error `ModuleNotFoundError: No module named 'src'`**:
   - Run `$env:PYTHONPATH="."` (Windows) or `export PYTHONPATH="."` (Linux) after activating the virtual environment.
2. **Error `ValueError: Incompatible Language version 15`**:
   - You cloned the latest tree-sitter grammar. Delete `vendor/tree-sitter-c` and `vendor/tree-sitter-cpp`, then re-clone using `--branch v0.23.0` as shown in Step 5.4.
3. **Training is extremely slow (e.g., 35s/batch)**:
   - Check if CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`.
   - If `False`, install the CUDA version of PyTorch: `python -m pip uninstall torch -y` and `python -m pip install torch --index-url https://download.pytorch.org/whl/cu121`.
4. **`RobertaModel ... newly initialized` warning**:
   - This is normal. We use the `CLS` token from `last_hidden_state`, so the randomly initialized `pooler` layer does not affect our results.

## 11. License
This project is for academic and research purposes. 

## 12. Contact
- **Group Members**: Hồ Lê Viết Nin, Trần Nhật Vinh, Ngô Văn Hiếu, Trịnh Quang Tin
- **Email**: hlvnin88@gmail.com