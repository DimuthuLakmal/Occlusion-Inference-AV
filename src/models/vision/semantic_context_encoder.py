import torch
import torch.nn as nn

from src.models.vision.convnext_map_encoder import ConvNeXtMapEncoder
from src.models.vision.cell_map_sampler import CellConditionedMultiScaleSampler
from src.models.vision.regnet_map_encoder import RegNetY800MFMapEncoder
from src.models.vision.cell_node_encoder import CellNodeEncoder


class SemanticContextEncoder(nn.Module):

    def __init__(
        self,
        num_semantic_classes: int,
        architecture: str = "regnet_y_800mf",
        map_context_dim: int = 64,
        cell_node_dim: int = 64,
        projection_dim: int = 32,
        pretrained: bool = True,
        stem_init: str = "random",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        # Select map backbone.
        if architecture == "convnext_tiny":

            self.map_encoder = (
                ConvNeXtMapEncoder(
                    in_channels=(
                        num_semantic_classes
                    ),
                    pretrained=pretrained,
                    stem_init=stem_init,
                    freeze_backbone=(
                        freeze_backbone
                    ),
                )
            )

        elif architecture == "regnet_y_800mf":

            self.map_encoder = (
                RegNetY800MFMapEncoder(
                    in_channels=(
                        num_semantic_classes
                    ),
                    pretrained=pretrained,
                    stem_init=stem_init,
                    freeze_backbone=(
                        freeze_backbone
                    ),
                )
            )

        self.sampler = (
            CellConditionedMultiScaleSampler(
                in_channels=self.map_encoder.out_channels,
                projection_dim=projection_dim,
                output_dim=map_context_dim,
                dropout=dropout,
            )
        )

        self.cell_encoder = (
            CellNodeEncoder(
                map_context_dim=map_context_dim,
                output_dim=cell_node_dim,
                dropout=dropout,
            )
        )

        self.mask_encoder = (
            CellNodeEncoder(
                map_context_dim=map_context_dim,
                output_dim=cell_node_dim,
                dropout=dropout,
            )
        )

    def forward(
        self,
        semantic_map: torch.Tensor,
        cell_xy: torch.Tensor,
        cell_mask=None,
    ):
        """
        semantic_map:
            [B,K,H,W]

        cell_xy:
            [B,Nc,2]
        """

        pyramid = self.map_encoder(semantic_map)

        map_context = self.sampler(
            pyramid=pyramid,
            cell_xy=cell_xy,
            cell_mask=cell_mask,
        )

        cell_embedding = (
            self.cell_encoder(
                cell_xy=cell_xy,
                map_context=map_context,
            )
        )

        z_mask_embedding = (
            self.mask_encoder(
                cell_xy=cell_xy,
                map_context=map_context,
            )
        )

        return {
            "cell_embedding": cell_embedding,
            "map_context": map_context,
            "map_pyramid": pyramid,
            "z_mask_embedding": z_mask_embedding,
        }