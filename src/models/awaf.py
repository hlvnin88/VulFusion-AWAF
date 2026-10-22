import torch
import torch.nn as nn


class AWAFModule(nn.Module):
    """
    Adaptive Weighted-Attention Fusion — Eq.(7),(8),(9):

        α = σ(W [E_C ; E_G] + b) ∈ [0,1]     (scalar per sample)
        E_fusion = α · E_C + (1 - α) · E_G

    W ∈ R^{2D×1}, b ∈ R
    """

    def __init__(self, emb_dim=768):
        super().__init__()
        self.proj = nn.Linear(emb_dim * 2, 1)

    def forward(self, e_c: torch.Tensor, e_g: torch.Tensor):
        h = torch.cat([e_c, e_g], dim=-1)          # (B, 2D)
        alpha = torch.sigmoid(self.proj(h))        # (B, 1)
        fused = alpha * e_c + (1.0 - alpha) * e_g  # broadcast
        return fused, alpha.squeeze(-1)