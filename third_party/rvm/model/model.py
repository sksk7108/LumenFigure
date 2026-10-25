"""Robust Video Matting network, slimmed to MobileNetV3 inference.

Original source: https://github.com/PeterL1n/RobustVideoMatting
License: GPL-3.0
The ResNet50 variant was removed because LumenFigure only vendors the
lightweight MobileNetV3 checkpoint.
"""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .decoder import Projection, RecurrentDecoder
from .deep_guided_filter import DeepGuidedFilterRefiner
from .fast_guided_filter import FastGuidedFilterRefiner
from .lraspp import LRASPP
from .mobilenetv3 import MobileNetV3LargeEncoder


class MattingNetwork(nn.Module):
    def __init__(
        self,
        variant: str = "mobilenetv3",
        refiner: str = "deep_guided_filter",
        pretrained_backbone: bool = False,
    ):
        super().__init__()
        assert variant == "mobilenetv3", "LumenFigure only vendors the MobileNetV3 RVM variant."
        assert refiner in ["fast_guided_filter", "deep_guided_filter"]

        self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
        self.aspp = LRASPP(960, 128)
        self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == "deep_guided_filter":
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()

    def forward(
        self,
        src: Tensor,
        r1: Optional[Tensor] = None,
        r2: Optional[Tensor] = None,
        r3: Optional[Tensor] = None,
        r4: Optional[Tensor] = None,
        downsample_ratio: float = 1,
        segmentation_pass: bool = False,
    ):
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src

        f1, f2, f3, f4 = self.backbone(src_sm)
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)

        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid)
            fgr = (fgr_residual + src).clamp(0.0, 1.0)
            pha = pha.clamp(0.0, 1.0)
            return [fgr, pha, *rec]

        seg = self.project_seg(hid)
        return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(
                x.flatten(0, 1),
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(
                x,
                scale_factor=scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        return x
