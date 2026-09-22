import torch
import torch.nn as nn


class CellNodeEncoder(nn.Module):
    """
    Create the cell-node representation from:

        normalized (x,y)
             +
        semantic map context
    """

    def __init__(
        self,
        map_context_dim: int = 64,
        coord_embedding_dim: int = 16,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.coord_encoder = (
            nn.Sequential(
                nn.Linear(
                    2,
                    coord_embedding_dim,
                ),
                nn.LayerNorm(
                    coord_embedding_dim
                ),
                nn.GELU(),

                nn.Linear(
                    coord_embedding_dim,
                    coord_embedding_dim,
                ),
            )
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                map_context_dim
                + coord_embedding_dim,
                output_dim,
            ),
            nn.LayerNorm(
                output_dim
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                output_dim,
                output_dim,
            ),
            nn.LayerNorm(
                output_dim
            ),
        )

    def forward(
        self,
        cell_xy: torch.Tensor,
        map_context: torch.Tensor,
    ):
        """
        cell_xy:
            [B,Nc,2]

        map_context:
            [B,Nc,D_map]

        returns:
            [B,Nc,D]
        """

        coord_embedding = (
            self.coord_encoder(
                cell_xy
            )
        )

        x = torch.cat(
            [
                coord_embedding,
                map_context,
            ],
            dim=-1,
        )

        return self.fusion(x)