import torch
import torch.nn as nn


class ResidualGatedFusion(nn.Module):
    """
    Residually inject map context into the graph representation.

    graph_feat:
        [B, Nc, D_graph]

    map_feat:
        [B, Nc, D_map]

    output:
        [B, Nc, D_graph]
    """

    def __init__(
        self,
        graph_dim: int,
        map_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        gate_bias: float = -2.0,
    ):
        super().__init__()

        self.graph_norm = nn.LayerNorm(graph_dim)
        self.map_norm = nn.LayerNorm(map_dim)

        # Map representation -> graph latent dimension
        self.map_proj = nn.Linear(
            map_dim,
            graph_dim,
        )

        # Learn the actual residual update
        self.update_net = nn.Sequential(
            nn.Linear(
                graph_dim * 2,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                graph_dim,
            ),
        )

        # Feature-wise/vector gate
        self.gate_net = nn.Linear(
            graph_dim * 2,
            graph_dim,
        )

        # Start with map contribution relatively small.
        nn.init.constant_(
            self.gate_net.bias,
            gate_bias,
        )

        nn.init.zeros_(
            self.update_net[-1].weight
        )

        nn.init.zeros_(
            self.update_net[-1].bias
        )

    def forward(
        self,
        graph_feat,
        map_feat,
    ):
        h_g = self.graph_norm(graph_feat)
        h_m = self.map_norm(map_feat)
        h_m = self.map_proj(h_m)
        # [B, Nc, Dg]

        joint = torch.cat([h_g, h_m], dim=-1,)
        # [B, Nc, 2*Dg]

        # Map-conditioned correction
        delta = self.update_net(joint)

        # Feature-wise gate
        gate = torch.sigmoid(self.gate_net(joint))
        # [B, Nc, Dg]

        # graph representation is never replaced.
        fused = (graph_feat + gate * delta)

        return fused, gate