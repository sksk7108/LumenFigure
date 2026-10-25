from __future__ import annotations

from pathlib import Path

import torch

from lumenfigure.paths import add_vendor_to_syspath


def auto_downsample_ratio(height: int, width: int) -> float:
    """Keep RVM's internal long side around 512px."""
    return min(512 / max(height, width), 1.0)


class PersonMatting:
    """Stateful RVM wrapper. Recurrent memory is kept across frames."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        add_vendor_to_syspath()
        from model import MattingNetwork

        self.device = device
        self.dtype = dtype
        self.model = MattingNetwork(variant="mobilenetv3").eval()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(device=device, dtype=dtype)
        self.reset()

    def reset(self) -> None:
        self.rec: list = [None] * 4

    @torch.inference_mode()
    def infer(self, src: torch.Tensor, downsample_ratio: float | None = None) -> torch.Tensor:
        """
        Args:
            src: RGB tensor in [0, 1], shape [1, 3, H, W].
        Returns:
            Alpha matte [1, 1, H, W] in [0, 1].
        """
        if downsample_ratio is None:
            downsample_ratio = auto_downsample_ratio(*src.shape[-2:])
        src = src.to(device=self.device, dtype=self.dtype)
        _fgr, pha, *self.rec = self.model(src, *self.rec, downsample_ratio=downsample_ratio)
        return pha
