import torch
import torch.nn as nn


class AWAFModule(nn.Module):
    def __init__(self, emb_dim=768):
        super().__init__()
        self.proj = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, e_c: torch.Tensor, e_g: torch.Tensor):
        alpha = torch.sigmoid(self.proj(torch.cat([e_c, e_g], dim=-1)))
        fused = alpha * e_c + (1 - alpha) * e_g
        return fused, alpha