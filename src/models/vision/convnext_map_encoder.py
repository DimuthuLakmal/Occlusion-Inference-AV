from collections import OrderedDict
from typing import Dict, Literal

import torch
import torch.nn as nn

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


class ConvNeXtMapEncoder(nn.Module):
    """
    Hierarchical map encoder based on ConvNeXt-Tiny.

    Input
    -----
    semantic_map:
        [B, K, H, W]

        K = number of semantic classes.

    Output
    ------
    Dict of spatial feature maps:

        s4:
            [B,  96, H/4,  W/4]

        s8:
            [B, 192, H/8,  W/8]

        s16:
            [B, 384, H/16, W/16]

        s32:
            [B, 768, H/32, W/32]
    """

    def __init__(
        self,
        in_channels: int,
        pretrained: bool = True,
        stem_init: Literal[
            "random",
            "rgb_mean",
        ] = "random",
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = (
            ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = convnext_tiny(
            weights=weights
        )

        if in_channels != 3:
            self._replace_input_stem(
                model=model,
                in_channels=in_channels,
                initialization=stem_init,
            )

        self.features = model.features

        self.out_channels = OrderedDict({
            "s4": 96,
            "s8": 192,
            "s16": 384,
            "s32": 768,
        })

        if freeze_backbone:
            for parameter in (
                self.features.parameters()
            ):
                parameter.requires_grad = False

    @staticmethod
    def _replace_input_stem(
        model: nn.Module,
        in_channels: int,
        initialization: str = "random",
    ):
        """
        Replace ConvNeXt's RGB input convolution.

        Original:
            3 -> 96

        New:
            K semantic channels -> 96

        I recommend random initialization for one-hot semantic
        inputs because semantic channels do not correspond to RGB.

        rgb_mean remains available as an ablation.
        """

        old_conv = model.features[0][0]

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
        )

        with torch.no_grad():

            if initialization == "random":

                # ConvNeXt-like initialization
                nn.init.trunc_normal_(
                    new_conv.weight,
                    std=0.02,
                )

                if new_conv.bias is not None:
                    nn.init.zeros_(
                        new_conv.bias
                    )

            elif initialization == "rgb_mean":

                # Convert pretrained RGB filters
                # [96,3,4,4] -> [96,1,4,4]
                mean_weight = (
                    old_conv.weight
                    .mean(
                        dim=1,
                        keepdim=True,
                    )
                )

                # [96,1,4,4]
                # ->
                # [96,K,4,4]
                new_conv.weight.copy_(
                    mean_weight.repeat(
                        1,
                        in_channels,
                        1,
                        1,
                    )
                )

                if new_conv.bias is not None:

                    if old_conv.bias is not None:
                        new_conv.bias.copy_(
                            old_conv.bias
                        )
                    else:
                        new_conv.bias.zero_()

            else:
                raise ValueError(
                    "initialization must be "
                    "'random' or 'rgb_mean'"
                )

        model.features[0][0] = new_conv

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        if x.ndim != 4:
            raise ValueError(
                "Expected semantic map "
                "[B, C, H, W]."
            )

        outputs = OrderedDict()

        # -------------------------------------------------
        # ConvNeXt stem
        #
        # 224 x 224
        #      ↓
        # 56 x 56
        # -------------------------------------------------
        x = self.features[0](x)

        # Stage 1
        x = self.features[1](x)

        outputs["s4"] = x

        # -------------------------------------------------
        # Downsample
        # 56 -> 28
        # -------------------------------------------------
        x = self.features[2](x)

        # Stage 2
        x = self.features[3](x)

        outputs["s8"] = x

        # -------------------------------------------------
        # Downsample
        # 28 -> 14
        # -------------------------------------------------
        x = self.features[4](x)

        # Stage 3
        x = self.features[5](x)

        outputs["s16"] = x

        # -------------------------------------------------
        # Downsample
        # 14 -> 7
        # -------------------------------------------------
        x = self.features[6](x)

        # Stage 4
        x = self.features[7](x)

        outputs["s32"] = x

        return outputs