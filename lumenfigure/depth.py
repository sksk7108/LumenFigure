from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from lumenfigure.paths import add_vendor_to_syspath

logging.getLogger("dinov2").setLevel(logging.ERROR)

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}


class DepthEstimator:
    """Video Depth Anything streaming wrapper.

    Uses the official sliding temporal cache so depth stays consistent across
    frames, while the ViT still sees one frame at a time (fits a 6GB GPU).
    Returned depth is the network-resolution map, larger = closer.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: torch.device,
        dtype: torch.dtype,
        encoder: str = "vits",
        input_size: int = 518,
    ) -> None:
        add_vendor_to_syspath()
        from video_depth_anything.video_depth_stream import INFER_LEN, VideoDepthAnything

        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported depth encoder: {encoder}")

        self.device = device
        self.dtype = dtype
        self.input_size = input_size
        self._infer_len = INFER_LEN
        self.fp32 = dtype != torch.float16 or device.type != "cuda"

        self.model = VideoDepthAnything(**MODEL_CONFIGS[encoder])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device=device).eval()

    def set_input_size(self, input_size: int) -> None:
        self.input_size = input_size

    def _autocast(self):
        enabled = (not self.fp32) and self.device.type == "cuda"
        return torch.autocast(device_type=self.device.type, enabled=enabled)

    def _prepare_transform(self, frame: np.ndarray) -> int:
        import cv2
        from torchvision.transforms import Compose

        from video_depth_anything.util.transform import NormalizeImage, PrepareForNet, Resize

        frame_height, frame_width = frame.shape[:2]
        self.model.frame_height = frame_height
        self.model.frame_width = frame_width
        input_size = self.input_size
        ratio = max(frame_height, frame_width) / min(frame_height, frame_width)
        if ratio > 1.78:
            input_size = int(input_size * 1.777 / ratio)
            input_size = round(input_size / 14) * 14
        self.model.transform = Compose(
            [
                Resize(
                    width=input_size,
                    height=input_size,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method="lower_bound",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ]
        )
        return input_size

    def _to_input(self, frame: np.ndarray) -> torch.Tensor:
        image = self.model.transform({"image": frame.astype(np.float32) / 255.0})["image"]
        return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)

    def _advance_window(self) -> None:
        self.model.frame_id_list.append(self.model.id)
        if self.model.id + self._infer_len > self.model.gap + 1:
            del self.model.frame_id_list[1]
            del self.model.frame_cache_list[1]

    @torch.inference_mode()
    def infer_frame(self, rgb: np.ndarray) -> torch.Tensor:
        """
        Args:
            rgb: uint8 RGB image, HWC. Size must stay constant for one video.
        Returns:
            Relative depth at network resolution, shape [h', w'], on self.device.
        """
        model = self.model
        model.id += 1

        if model.transform is None:
            self._prepare_transform(rgb)
            cur_input = self._to_input(rgb)
            with self._autocast():
                features = model.forward_features(cur_input)
                depth, cached = model.forward_depth(features, cur_input.shape)
            model.frame_cache_list = [cached] * self._infer_len
            model.frame_id_list.extend([0] * (self._infer_len - 1))
            out = depth[0, 0]
        else:
            frame_height, frame_width = rgb.shape[:2]
            if frame_height != model.frame_height or frame_width != model.frame_width:
                raise ValueError("Video Depth Anything streaming expects a constant frame size.")
            cur_input = self._to_input(rgb)
            with self._autocast():
                features = model.forward_features(cur_input)
            cur_list = model.frame_cache_list[0:2] + model.frame_cache_list[-self._infer_len + 3 :]
            if len(cur_list) != self._infer_len - 1:
                raise RuntimeError(f"Temporal cache length {len(cur_list)} != {self._infer_len - 1}")
            cur_cache = [torch.cat([h[i] for h in cur_list], dim=1) for i in range(len(cur_list[0]))]
            with self._autocast():
                depth, new_cache = model.forward_depth(
                    features,
                    cur_input.shape,
                    cached_hidden_state_list=cur_cache,
                )
            model.frame_cache_list.append(new_cache)
            out = depth[0, 0]

        self._advance_window()
        return out.float()
