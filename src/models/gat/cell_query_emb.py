import torch
import torch.nn as nn

class CellQueryEmb(nn.Module):
    """
    Inputs
      cell_xy_norm : [B, N2, 2]  (x,y ∈ [-1,1] in the SAME image/map space used by your frames)
      cell_pad     : [B, N2] bool (True=pad)  optional

    Output
      Q_cell       : [B, N2, D]
    """
    def __init__(self, d_model=256, mode="fourier", n_freq=8, mlp_hidden=128, dropout=0.1):
        super().__init__()
        assert mode in {"mlp", "fourier"}
        self.mode = mode
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

        in_dim = 2

        self.mode_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

        # Small init on the last layer helps keep logits stable early on
        nn.init.trunc_normal_(self.mode_mlp[-1].weight, std=0.02)
        nn.init.zeros_(self.mode_mlp[-1].bias)

    def forward(self, cell_xy_norm, cell_pad=None):
        feats = cell_xy_norm                        # [B,N2,2]

        q = self.mode_mlp(feats)                        # [B,N2,D]
        q = self.ln(self.dropout(q))

        if cell_pad is not None:
            q = q.masked_fill(cell_pad.unsqueeze(-1), 0.0)
        return q