import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from src.dfg.extract import DFGExtractor
from src.graph_features import build_graphcodebert_inputs


class VulnDataset(Dataset):
    def __init__(
            self,
            rows,
            codebert_name: str,
            graphcodebert_name: str,
            tree_sitter_so: str,
            max_len_code: int = 256,
            max_len_graph: int = 256,
    ):
        self.rows = rows
        self.code_tok = AutoTokenizer.from_pretrained(codebert_name)
        self.graph_tok = AutoTokenizer.from_pretrained(graphcodebert_name)
        self.max_len_code = max_len_code
        self.max_len_graph = max_len_graph
        self.tree_sitter_so = tree_sitter_so
        self._dfg_cache = {}

        # LƯU Ý: Không khởi tạo DFGExtractor ở đây để tránh lỗi khi dùng num_workers > 0

    def __len__(self):
        return len(self.rows)

    def _get_dfg(self, code: str, lang: str):
        # Khởi tạo lazy (mỗi worker sẽ tự tạo 1 instance riêng)
        if not hasattr(self, "_extractor"):
            self._extractor = DFGExtractor(self.tree_sitter_so)

        key = hash(code)
        if key in self._dfg_cache:
            return self._dfg_cache[key]
        try:
            info = self._extractor.extract(code, lang)
        except Exception as e:
            # print(f"[WARN] DFG extraction failed: {e}") # Tắt print để không làm chậm log
            info = {"dfg": [], "token_names": []}

        # Giới hạn cache để tránh tràn RAM nếu dữ liệu quá lớn
        if len(self._dfg_cache) > 5000:
            self._dfg_cache.clear()

        self._dfg_cache[key] = info
        return info

    def __getitem__(self, idx):
        row = self.rows[idx]
        code = row["code"]
        label = int(row["label"])
        lang = row.get("language", "cpp")

        enc_c = self.code_tok(
            code, truncation=True, padding="max_length",
            max_length=self.max_len_code, return_tensors="pt",
        )

        dfg_info = self._get_dfg(code, lang)
        g = build_graphcodebert_inputs(
            self.graph_tok, code, dfg_info, max_len=self.max_len_graph
        )

        return {
            "input_ids_c": enc_c["input_ids"].squeeze(0),
            "attention_mask_c": enc_c["attention_mask"].squeeze(0),
            "input_ids_g": g["input_ids"],
            "attention_mask_g": g["attention_mask"],
            "position_idx_g": g["position_idx"],
            "label": torch.tensor(label, dtype=torch.long),
        }