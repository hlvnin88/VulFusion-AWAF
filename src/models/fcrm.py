import torch
import torch.nn as nn
from transformers import AutoModel
from .awaf import AWAFModule


class MLPClassifier(nn.Module):
    """FC + Softmax (Eq.10). Trả về logits (B, num_classes)."""

    def __init__(self, in_dim=768, hidden_dims=(512, 256), dropout=0.2, num_classes=2):
        super().__init__()
        layers, cur = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]
            cur = h
        layers += [nn.Linear(cur, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)                     # (B, 2)


def _encode_graph(encoder, input_ids_g, attention_mask_g, position_idx_g=None):
    kw = {"input_ids": input_ids_g, "attention_mask": attention_mask_g}
    if position_idx_g is not None:
        kw["position_ids"] = position_idx_g
    return encoder(**kw)


class FCRMBase(nn.Module):
    """
    Baseline FCRM: fusion CỐ ĐỊNH (α là tham số học toàn cục, giống nhau mọi mẫu).
    Khớp Mục III.D báo cáo: "FCRM adopts ... fixed feature fusion strategy".
    """

    def __init__(self, codebert_name, graphcodebert_name,
                 hidden_dim=768, dropout=0.2, mlp_hidden_dims=(512, 256)):
        super().__init__()
        self.code_encoder = AutoModel.from_pretrained(codebert_name)
        self.graph_encoder = AutoModel.from_pretrained(graphcodebert_name)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))    # sigmoid(0) = 0.5
        self.classifier = MLPClassifier(hidden_dim, tuple(mlp_hidden_dims),
                                        dropout, num_classes=2)

    @staticmethod
    def _cls(out):
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids_c, attention_mask_c,
                input_ids_g, attention_mask_g, position_idx_g=None):
        out_c = self.code_encoder(input_ids=input_ids_c, attention_mask=attention_mask_c)
        out_g = _encode_graph(self.graph_encoder, input_ids_g, attention_mask_g, position_idx_g)
        e_c, e_g = self._cls(out_c), self._cls(out_g)

        a = torch.sigmoid(self.alpha_raw)                   # scalar
        fused = a * e_c + (1.0 - a) * e_g
        return self.classifier(fused)


class FCRMAWAF(nn.Module):
    """Đề xuất: Adaptive Weighted-Attention Fusion (đóng góp chính)."""

    def __init__(self, codebert_name, graphcodebert_name,
                 hidden_dim=768, dropout=0.2, mlp_hidden_dims=(512, 256)):
        super().__init__()
        self.code_encoder = AutoModel.from_pretrained(codebert_name)
        self.graph_encoder = AutoModel.from_pretrained(graphcodebert_name)
        self.awaf = AWAFModule(hidden_dim)
        self.classifier = MLPClassifier(hidden_dim, tuple(mlp_hidden_dims),
                                        dropout, num_classes=2)

    @staticmethod
    def _cls(out):
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids_c, attention_mask_c,
                input_ids_g, attention_mask_g, position_idx_g=None):
        out_c = self.code_encoder(input_ids=input_ids_c, attention_mask=attention_mask_c)
        out_g = _encode_graph(self.graph_encoder, input_ids_g, attention_mask_g, position_idx_g)
        e_c, e_g = self._cls(out_c), self._cls(out_g)

        fused, _ = self.awaf(e_c, e_g)
        return self.classifier(fused)