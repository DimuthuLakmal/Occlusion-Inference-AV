from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.vision.layers import (
    LayerNorm2d,
)


class CellConditionedMultiScaleSampler(
    nn.Module
):
    """
    Query hierarchical semantic-map features at the
    coordinates of critical occluded cells.

    Inputs
    ------
    pyramid:
        {
            "s4":  [B, C1, H1, W1],
            "s8":  [B, C2, H2, W2],
            "s16": [B, C3, H3, W3],
            "s32": [B, C4, H4, W4]
        }

    cell_xy:
        [B, Nc, 2]

        Coordinates MUST be normalized:

            x_norm = x / (W_original - 1)
            y_norm = y / (H_original - 1)

        therefore x,y are in [0,1].

    cell_mask:
        Optional [B, Nc].

        True  -> valid critical cell
        False -> padded cell

    Output
    ------
    map_context:
        [B, Nc, output_dim]
    """

    def __init__(
        self,
        in_channels: Mapping[str, int],
        projection_dim: int = 32,
        output_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.stage_names = list(
            in_channels.keys()
        )

        # --------------------------------------------------
        # Convert every scale to the same feature dimension.
        #
        # 96  -> 32
        # 192 -> 32
        # 384 -> 32
        # 768 -> 32
        # --------------------------------------------------
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=projection_dim,
                    kernel_size=1,
                    bias=False,
                ),
                LayerNorm2d(
                    projection_dim
                ),
                nn.GELU(),
            )
            for name, channels
            in in_channels.items()
        })

        fused_dim = (
            projection_dim
            * len(self.stage_names)
        )

        # --------------------------------------------------
        # Fuse sampled features from all ConvNeXt scales.
        #
        # e.g.
        # 4 × 32 = 128
        #
        # 128 -> 64
        # --------------------------------------------------
        self.fusion = nn.Sequential(
            nn.Linear(
                fused_dim,
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

    @staticmethod
    def _normalized_xy_to_grid(
        cell_xy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert [0,1] coordinates to the [-1,1]
        coordinate system expected by grid_sample.

        Example:
            (0.25, 0.30)
                 ↓
            (-0.50, -0.40)
        """

        return (
            2.0 * cell_xy - 1.0
        )

    @staticmethod
    def _validate_coordinates(
        cell_xy: torch.Tensor,
        cell_mask: torch.Tensor,
    ):
        valid_xy = cell_xy[
            cell_mask
        ]

        if valid_xy.numel() == 0:
            return

        if (
            (valid_xy < 0.0).any()
            or
            (valid_xy > 1.0).any()
        ):
            min_value = (
                valid_xy.min().item()
            )

            max_value = (
                valid_xy.max().item()
            )

            raise ValueError(
                "Valid cell coordinates must "
                "be in [0,1]. "
                f"Observed [{min_value}, "
                f"{max_value}]."
            )

    def forward(
        self,
        pyramid: Mapping[
            str,
            torch.Tensor,
        ],
        cell_xy: torch.Tensor,
        cell_mask: Optional[
            torch.Tensor
        ] = None,
        return_per_scale: bool = False,
    ):

        if (
            cell_xy.ndim != 3
            or
            cell_xy.shape[-1] != 2
        ):
            raise ValueError(
                "cell_xy must have shape "
                "[B, Nc, 2]."
            )

        batch_size, num_cells, _ = (
            cell_xy.shape
        )

        if cell_mask is None:
            cell_mask = torch.ones(
                batch_size,
                num_cells,
                dtype=torch.bool,
                device=cell_xy.device,
            )

        else:
            cell_mask = (
                cell_mask.bool()
            )

        # self._validate_coordinates(
        #     cell_xy,
        #     cell_mask,
        # )

        # --------------------------------------------------
        # For padded cells, temporarily place coordinate
        # at zero so grid_sample receives a valid location.
        # Their final features will be zeroed later.
        # --------------------------------------------------
        safe_xy = torch.where(
            cell_mask.unsqueeze(-1),
            cell_xy,
            torch.zeros_like(cell_xy),
        )

        # [0,1] -> [-1,1]
        sampling_grid = (
            self._normalized_xy_to_grid(
                safe_xy
            )
        )

        # grid_sample expects:
        #
        # [B, H_out, W_out, 2]
        #
        # We treat:
        #
        # H_out = Nc
        # W_out = 1
        #
        # giving:
        # [B, Nc, 1, 2]
        sampling_grid = (
            sampling_grid.unsqueeze(2)
        )

        sampled_features = []
        per_scale = {}

        for stage_name in (
            self.stage_names
        ):

            if stage_name not in pyramid:
                raise KeyError(
                    f"Missing pyramid stage: "
                    f"{stage_name}"
                )

            feature = (
                pyramid[stage_name]
            )

            # ----------------------------------------------
            # Channel projection.
            # ----------------------------------------------
            feature = (
                self.projections[
                    stage_name
                ](feature)
            )

            # ----------------------------------------------
            # Differentiably query each cell coordinate.
            #
            # Important:
            # align_corners=True matches coordinates defined
            # using x/(W-1), y/(H-1).
            # ----------------------------------------------
            sampled = F.grid_sample(
                input=feature,
                grid=sampling_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )

            # Result:
            #
            # [B, C_projected, Nc, 1]
            #
            # ->
            #
            # [B, Nc, C_projected]
            sampled = (
                sampled
                .squeeze(-1)
                .transpose(1, 2)
            )

            # Zero padded cells
            sampled = (
                sampled
                * cell_mask
                .unsqueeze(-1)
                .to(sampled.dtype)
            )

            sampled_features.append(
                sampled
            )

            per_scale[stage_name] = (
                sampled
            )

        # --------------------------------------------------
        # [B,Nc,32] × 4
        #
        # ->
        #
        # [B,Nc,128]
        # --------------------------------------------------
        multi_scale_features = (
            torch.cat(
                sampled_features,
                dim=-1,
            )
        )

        # --------------------------------------------------
        # [B,Nc,128]
        #
        # ->
        #
        # [B,Nc,64]
        # --------------------------------------------------
        map_context = self.fusion(
            multi_scale_features
        )

        map_context = (
            map_context
            * cell_mask
            .unsqueeze(-1)
            .to(map_context.dtype)
        )

        if return_per_scale:
            return (
                map_context,
                per_scale,
            )

        return map_context