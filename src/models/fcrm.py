import torch
import torch.nn as nn
from transformers import AutoModel
from .awaf import AWAFModule


class MLPClassifier(nn.Module):
    def __init__(self, in_dim=768, hidden_dims=(512, 256), dropout=0.2):
        super().__init__()
        layers, cur = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(cur, h), nn.ReLU(), nn.Dropout(dropout)]
            cur = h
        layers += [nn.Linear(cur, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FCRMBase(nn.Module):
    def __init__(self, codebert_name, graphcodebert_name, hidden_dim=768, dropout=0.2, mlp_hidden_dims=(512, 256)):
        super().__init__()
        self.code_encoder = AutoModel.from_pretrained(codebert_name)
        self.graph_encoder = AutoModel.from_pretrained(graphcodebert_name)
        self.alpha_raw = nn.Parameter(torch.tensor([0.5], dtype=torch.float))
        self.classifier = MLPClassifier(hidden_dim, mlp_hidden_dims, dropout)

    def _cls(self, out):
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids_c, attention_mask_c, input_ids_g, attention_mask_g, position_idx_g=None):
        out_c = self.code_encoder(input_ids=input_ids_c, attention_mask=attention_mask_c)
        out_g = self.graph_encoder(input_ids=input_ids_g, attention_mask=attention_mask_g)

        e_c = self._cls(out_c)
        e_g = self._cls(out_g)

        a = self.alpha_raw.sigmoid()
        fused = a * e_c + (1 - a) * e_g
        return self.classifier(fused)


class FCRMAWAF(nn.Module):
    def __init__(self, codebert_name, graphcodebert_name, hidden_dim=768, dropout=0.2, mlp_hidden_dims=(512, 256)):
        super().__init__()
        self.code_encoder = AutoModel.from_pretrained(codebert_name)
        self.graph_encoder = AutoModel.from_pretrained(graphcodebert_name)
        self.awaf = AWAFModule(hidden_dim)
        self.classifier = MLPClassifier(hidden_dim, mlp_hidden_dims, dropout)

    def _cls(self, out):
        return out.last_hidden_state[:, 0, :]

    def forward(self, input_ids_c, attention_mask_c, input_ids_g, attention_mask_g, position_idx_g=None):
        out_c = self.code_encoder(input_ids=input_ids_c, attention_mask=attention_mask_c)
        out_g = self.graph_encoder(input_ids=input_ids_g, attention_mask=attention_mask_g)

        e_c = self._cls(out_c)
        e_g = self._cls(out_g)

        fused, _ = self.awaf(e_c, e_g)
        return self.classifier(fused)