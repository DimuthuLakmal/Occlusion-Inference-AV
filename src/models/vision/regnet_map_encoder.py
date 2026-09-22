from collections import OrderedDict
from typing import Dict, Literal

import math

import torch
import torch.nn as nn

from torchvision.models import (
    regnet_y_800mf,
    RegNet_Y_800MF_Weights,
)


class RegNetY800MFMapEncoder(nn.Module):
    """
    Hierarchical RegNetY-800MF encoder for one-hot semantic maps.

    Input
    -----
    x:
        [B, K, H, W]

        K = number of semantic classes.

    Output
    ------
    {
        "s4":  [B,  64, H/4,  W/4],
        "s8":  [B, 128, H/8,  W/8],
        "s16": [B, 320, H/16, W/16],
        "s32": [B, 768, H/32, W/32],
    }
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
        freeze_bn_stats: bool = True,
    ):
        super().__init__()

        weights = (
            RegNet_Y_800MF_Weights.DEFAULT
            if pretrained
            else None
        )

        model = regnet_y_800mf(
            weights=weights
        )

        # -------------------------------------------------
        # RegNet expects RGB:
        #
        #     3 -> 32
        #
        # Replace first convolution so it accepts
        # K semantic one-hot channels:
        #
        #     K -> 32
        # -------------------------------------------------
        if in_channels != 3:
            self._replace_input_stem(
                model=model,
                in_channels=in_channels,
                initialization=stem_init,
            )

        self.stem = model.stem

        # TorchVision RegNet structure:
        #
        # stem
        # trunk_output.block1
        # trunk_output.block2
        # trunk_output.block3
        # trunk_output.block4
        #
        self.stage1 = (
            model.trunk_output.block1
        )

        self.stage2 = (
            model.trunk_output.block2
        )

        self.stage3 = (
            model.trunk_output.block3
        )

        self.stage4 = (
            model.trunk_output.block4
        )

        # RegNetY-800MF stage output dimensions
        self.out_channels = OrderedDict({
            "s4": 64,
            "s8": 144,
            "s16": 320,
            "s32": 784,
        })

        self.freeze_backbone = (
            freeze_backbone
        )

        self.freeze_bn_stats = (
            freeze_bn_stats
        )

        if self.freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        """
        Freeze RegNet stages but leave semantic stem trainable.

        This is useful for your current overfitting issue.
        """

        for stage in (
            self.stage1,
            self.stage2,
            self.stage3,
            self.stage4,
        ):
            for parameter in (
                stage.parameters()
            ):
                parameter.requires_grad = False

        # Keep semantic stem trainable.
        for parameter in (
            self.stem.parameters()
        ):
            parameter.requires_grad = True

    @staticmethod
    def _replace_input_stem(
        model: nn.Module,
        in_channels: int,
        initialization: str,
    ):
        """
        Replace RegNet's RGB input Conv2d.

        Original:
            Conv2d(3, 32, 3x3, stride=2)

        New:
            Conv2d(K, 32, 3x3, stride=2)

        For categorical one-hot semantic maps, random initialization
        is recommended because semantic channels have no RGB meaning.
        """

        old_conv = model.stem[0]

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

                # Match RegNet/ResNet-style Conv initialization.
                fan_out = (
                    new_conv.kernel_size[0]
                    * new_conv.kernel_size[1]
                    * new_conv.out_channels
                )

                nn.init.normal_(
                    new_conv.weight,
                    mean=0.0,
                    std=math.sqrt(
                        2.0 / fan_out
                    ),
                )

                if new_conv.bias is not None:
                    nn.init.zeros_(
                        new_conv.bias
                    )

            elif initialization == "rgb_mean":

                # Pretrained RGB weights:
                #
                # [32, 3, 3, 3]
                #
                # ->
                #
                # [32, 1, 3, 3]
                mean_weight = (
                    old_conv.weight
                    .mean(
                        dim=1,
                        keepdim=True,
                    )
                )

                # ->
                #
                # [32, K, 3, 3]
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
                    "'random' or 'rgb_mean'."
                )

        model.stem[0] = new_conv

    def train(
        self,
        mode: bool = True,
    ):
        """
        If the RegNet trunk is frozen, optionally keep its
        BatchNorm running statistics frozen as well.

        Merely setting requires_grad=False does NOT stop
        BatchNorm running mean/variance updates.
        """

        super().train(mode)

        if (
            mode
            and self.freeze_backbone
            and self.freeze_bn_stats
        ):
            self.stage1.eval()
            self.stage2.eval()
            self.stage3.eval()
            self.stage4.eval()

        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:

        if x.ndim != 4:
            raise ValueError(
                "Expected input with shape "
                "[B, C, H, W]."
            )

        outputs = OrderedDict()

        # -------------------------------------------------
        # Input:
        #
        # [B,K,224,224]
        #
        # Stem has stride 2:
        #
        # -> [B,32,112,112]
        # -------------------------------------------------
        x = self.stem(x)

        # -------------------------------------------------
        # Stage 1:
        #
        # 112 -> 56
        # -------------------------------------------------
        x = self.stage1(x)

        outputs["s4"] = x

        # [B,64,56,56]

        # -------------------------------------------------
        # Stage 2:
        #
        # 56 -> 28
        # -------------------------------------------------
        x = self.stage2(x)

        outputs["s8"] = x

        # [B,128,28,28]

        # -------------------------------------------------
        # Stage 3:
        #
        # 28 -> 14
        # -------------------------------------------------
        x = self.stage3(x)

        outputs["s16"] = x

        # [B,320,14,14]

        # -------------------------------------------------
        # Stage 4:
        #
        # 14 -> 7
        # -------------------------------------------------
        x = self.stage4(x)

        outputs["s32"] = x

        # [B,768,7,7]

        return outputs