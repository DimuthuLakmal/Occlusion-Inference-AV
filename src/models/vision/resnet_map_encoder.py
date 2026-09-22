from collections import OrderedDict
from typing import Dict, Literal

import math

import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class ResNet18MapEncoder(nn.Module):
    """
    Hierarchical ResNet-18 encoder for one-hot semantic maps.

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
        "s16": [B, 256, H/16, W/16],
        "s32": [B, 512, H/32, W/32],
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
            ResNet18_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        model = resnet18(
            weights=weights
        )

        # -------------------------------------------------
        # ResNet expects RGB:
        #
        #     3 -> 64
        #
        # Replace first convolution so it accepts
        # K semantic one-hot channels:
        #
        #     K -> 64
        # -------------------------------------------------
        if in_channels != 3:
            self._replace_input_stem(
                model=model,
                in_channels=in_channels,
                initialization=stem_init,
            )

        self.stem = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
        )

        # TorchVision ResNet structure:
        #
        # conv1 + bn1 + relu + maxpool  (stride 4)
        # layer1  (stride 1  -> s4)
        # layer2  (stride 2  -> s8)
        # layer3  (stride 2  -> s16)
        # layer4  (stride 2  -> s32)
        #
        self.stage1 = model.layer1
        self.stage2 = model.layer2
        self.stage3 = model.layer3
        self.stage4 = model.layer4

        # ResNet-18 stage output dimensions
        self.out_channels = OrderedDict({
            "s4": 64,
            "s8": 128,
            "s16": 256,
            "s32": 512,
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
        Freeze ResNet stages but leave semantic stem trainable.
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
        Replace ResNet's RGB input Conv2d.

        Original:
            Conv2d(3, 64, 7x7, stride=2)

        New:
            Conv2d(K, 64, 7x7, stride=2)

        For categorical one-hot semantic maps, random initialization
        is recommended because semantic channels have no RGB meaning.
        """

        old_conv = model.conv1

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

                # Match ResNet-style Conv initialization.
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
                # [64, 3, 7, 7]
                #
                # ->
                #
                # [64, 1, 7, 7]
                mean_weight = (
                    old_conv.weight
                    .mean(
                        dim=1,
                        keepdim=True,
                    )
                )

                # ->
                #
                # [64, K, 7, 7]
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

        model.conv1 = new_conv

    def train(
        self,
        mode: bool = True,
    ):
        """
        If the ResNet trunk is frozen, optionally keep its
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
        # Stem (conv1 stride2 + maxpool stride2):
        #
        # -> [B,64,56,56]
        # -------------------------------------------------
        x = self.stem(x)

        # -------------------------------------------------
        # Stage 1:
        #
        # 56 -> 56 (stride 1)
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

        # [B,256,14,14]

        # -------------------------------------------------
        # Stage 4:
        #
        # 14 -> 7
        # -------------------------------------------------
        x = self.stage4(x)

        outputs["s32"] = x

        # [B,512,7,7]

        return outputs
