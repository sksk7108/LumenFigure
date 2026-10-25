from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from lumenfigure.depth import DepthEstimator
from lumenfigure.matting import PersonMatting
from lumenfigure.paths import DEFAULT_DEPTH_WEIGHTS, DEFAULT_RVM_WEIGHTS
from lumenfigure.silhouette import SilhouetteMapper
from lumenfigure.video_io import AsyncLumaWriter, VideoReader, fit_max_side


@dataclass
class PipelineConfig:
    input_path: Path
    output_path: Path
    rvm_weights: Path = DEFAULT_RVM_WEIGHTS
    depth_weights: Path = DEFAULT_DEPTH_WEIGHTS
    device: str = "auto"
    fp16: bool = True
    depth_size: int = 518
    depth_stride: int = 1
    max_side: int | None = 1920
    use_matte: bool = True
    gamma: float = 0.85
    contrast: float = 1.0
    min_luma: float = 0.12
    max_luma: float = 1.0
    temporal_smooth: bool = True
    blur_depth: bool = False
    blur_kernel: int = 5


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resize_hw(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if image.shape[-2] == height and image.shape[-1] == width:
        return image
    return F.interpolate(image[None, None], size=(height, width), mode="bilinear", align_corners=True)[0, 0]


class FrameUploader:
    """Pin a CPU staging buffer so H2D copies can overlap with compute."""

    def __init__(self, device: torch.device, dtype: torch.dtype) -> None:
        self.device = device
        self.dtype = dtype
        self.pinned: torch.Tensor | None = None

    def upload_bgr(self, bgr: np.ndarray) -> torch.Tensor:
        src = torch.from_numpy(np.ascontiguousarray(bgr))
        if self.device.type == "cuda":
            if self.pinned is None or self.pinned.shape != src.shape:
                self.pinned = torch.empty(src.shape, dtype=torch.uint8, pin_memory=True)
            self.pinned.copy_(src)
            pixel = self.pinned.to(self.device, non_blocking=True)
        else:
            pixel = src.to(self.device)
        rgb = pixel[..., [2, 1, 0]].permute(2, 0, 1).contiguous()
        return rgb.unsqueeze(0).to(dtype=self.dtype).div_(255.0)


def _build_models(
    cfg: PipelineConfig,
    device: torch.device,
    dtype: torch.dtype,
    depth_size: int,
    depth_device: torch.device | None = None,
) -> tuple[PersonMatting | None, DepthEstimator, SilhouetteMapper]:
    matting = None
    if cfg.use_matte:
        matting = PersonMatting(cfg.rvm_weights, device=device, dtype=dtype)
    depth = DepthEstimator(
        cfg.depth_weights,
        device=depth_device or device,
        dtype=dtype if (depth_device or device).type == "cuda" else torch.float32,
        encoder="vits",
        input_size=depth_size,
    )
    mapper = SilhouetteMapper(
        gamma=cfg.gamma,
        contrast=cfg.contrast,
        min_luma=cfg.min_luma,
        max_luma=cfg.max_luma,
        temporal_smooth=cfg.temporal_smooth,
        blur_depth=cfg.blur_depth,
        blur_kernel=cfg.blur_kernel,
    )
    return matting, depth, mapper


def _convert_once(
    cfg: PipelineConfig,
    device: torch.device,
    dtype: torch.dtype,
    depth_size: int,
    depth_device: torch.device | None = None,
) -> None:
    matting, depth, mapper = _build_models(cfg, device, dtype, depth_size, depth_device)
    uploader = FrameUploader(device, dtype)

    with VideoReader(cfg.input_path) as reader:
        out_h, out_w = fit_max_side(reader.height, reader.width, cfg.max_side)
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        with AsyncLumaWriter(cfg.output_path, reader.fps, out_w, out_h, device) as writer:
            total = reader.frame_count if reader.frame_count > 0 else None
            for _index, bgr in enumerate(tqdm(reader.frames(), total=total, desc="LumenFigure", unit="frame")):
                if bgr.shape[0] != out_h or bgr.shape[1] != out_w:
                    bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)

                src = uploader.upload_bgr(bgr)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                if matting is not None:
                    alpha = matting.infer(src)[0, 0].float()
                else:
                    alpha = torch.ones(out_h, out_w, device=device, dtype=torch.float32)

                depth_map = depth.infer_frame(rgb)
                if depth_map.device != alpha.device:
                    depth_map = depth_map.to(alpha.device, non_blocking=True)

                alpha_low = resize_hw(alpha, depth_map.shape[0], depth_map.shape[1])
                luma_low = mapper.map(depth_map, alpha_low)
                luma = resize_hw(luma_low, out_h, out_w) * alpha.clamp(0.0, 1.0)
                writer.write_luma(luma)

    if device.type == "cuda":
        torch.cuda.empty_cache()


def convert_video(cfg: PipelineConfig) -> None:
    if cfg.use_matte and not cfg.rvm_weights.is_file():
        raise FileNotFoundError(f"RVM weights not found: {cfg.rvm_weights}")
    if not cfg.depth_weights.is_file():
        raise FileNotFoundError(f"Depth weights not found: {cfg.depth_weights}")
    if cfg.depth_stride != 1:
        print("Video Depth Anything streaming uses every frame; --depth-stride is ignored.")
    if cfg.contrast <= 0:
        raise ValueError("--contrast must be > 0")
    if cfg.gamma <= 0:
        raise ValueError("--gamma must be > 0")
    if not 0.0 <= cfg.min_luma < cfg.max_luma <= 1.0:
        raise ValueError("--shadow/--min-luma must be < --highlight/--max-luma, both in [0, 1]")

    device = resolve_device(cfg.device)
    dtype = torch.float16 if (cfg.fp16 and device.type == "cuda") else torch.float32
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    candidates: list[int] = [cfg.depth_size]
    for fallback in (392, 280):
        if cfg.depth_size > fallback and fallback not in candidates:
            candidates.append(fallback)

    last_error: BaseException | None = None
    for depth_size in candidates:
        try:
            _convert_once(cfg, device, dtype, depth_size, None)
            return
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            torch.cuda.empty_cache()
            print(f"CUDA OOM at depth_size={depth_size}, retrying with a smaller input...")

    raise RuntimeError("Failed to convert video within available GPU memory.") from last_error
