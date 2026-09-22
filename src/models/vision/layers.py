import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    Apply LayerNorm over channels independently
    at each spatial location.

    Input:
        [B, C, H, W]

    LayerNorm operates on:
        [C]

    independently for each:
        [B, H, W]
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            num_channels,
            eps=eps,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # [B, C, H, W]
        # ->
        # [B, H, W, C]
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)

        # [B, H, W, C]
        # ->
        # [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        return x