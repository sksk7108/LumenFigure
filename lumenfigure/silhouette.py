from __future__ import annotations

import torch
import torch.nn.functional as F


class SilhouetteMapper:
    """Turn relative depth + alpha into a flat near-bright / far-dark luma map."""

    def __init__(
        self,
        gamma: float = 0.85,
        contrast: float = 1.0,
        min_luma: float = 0.12,
        max_luma: float = 1.0,
        far_percentile: float = 0.02,
        near_percentile: float = 0.98,
        alpha_threshold: float = 0.05,
        temporal_smooth: bool = True,
        depth_ema: float = 0.3,
        stats_ema: float = 0.2,
        blur_depth: bool = False,
        blur_kernel: int = 5,
    ) -> None:
        self.gamma = gamma
        self.contrast = contrast
        self.min_luma = min_luma
        self.max_luma = max_luma
        self.far_percentile = far_percentile
        self.near_percentile = near_percentile
        self.alpha_threshold = alpha_threshold
        self.temporal_smooth = temporal_smooth
        self.depth_ema = depth_ema
        self.stats_ema = stats_ema
        self.blur_depth = blur_depth
        self.blur_kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        self._p_far: torch.Tensor | None = None
        self._p_near: torch.Tensor | None = None
        self._prev_depth: torch.Tensor | None = None

    def reset(self) -> None:
        self._p_far = None
        self._p_near = None
        self._prev_depth = None

    def _maybe_blur(self, depth: torch.Tensor) -> torch.Tensor:
        if not self.blur_depth:
            return depth
        k = self.blur_kernel
        padding = k // 2
        kernel = torch.ones(1, 1, k, k, device=depth.device, dtype=torch.float32)
        kernel = kernel / kernel.numel()
        return F.conv2d(depth[None, None], kernel, padding=padding)[0, 0]

    def map(self, depth: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Args:
            depth: [H, W] relative depth, larger = closer.
            alpha: [H, W] person matte in [0, 1].
        Returns:
            [H, W] unpremultiplied luma in [min_luma, max_luma]. Background is
            composed later with the full-resolution alpha.
        """
        depth = depth.float()
        alpha = alpha.float()
        depth = self._maybe_blur(depth)

        if self.temporal_smooth and self._prev_depth is not None:
            depth = (1.0 - self.depth_ema) * depth + self.depth_ema * self._prev_depth
        self._prev_depth = depth

        mask = alpha > self.alpha_threshold
        if not mask.any():
            return torch.zeros_like(depth)

        values = depth[mask]
        p_far = torch.quantile(values, self.far_percentile)
        p_near = torch.quantile(values, self.near_percentile)

        if self.temporal_smooth and self._p_far is not None:
            p_far = (1.0 - self.stats_ema) * self._p_far + self.stats_ema * p_far
            p_near = (1.0 - self.stats_ema) * self._p_near + self.stats_ema * p_near

        self._p_far = p_far
        self._p_near = p_near

        span = (p_near - p_far).clamp_min(1e-6)
        norm = ((depth - p_far) / span).clamp(0.0, 1.0)
        if self.contrast != 1.0:
            norm = ((norm - 0.5) * self.contrast + 0.5).clamp(0.0, 1.0)
        luma = norm.pow(self.gamma)
        luma = luma * (self.max_luma - self.min_luma) + self.min_luma
        return luma
