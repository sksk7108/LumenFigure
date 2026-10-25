from __future__ import annotations

import shutil
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import torch


def even_size(height: int, width: int) -> tuple[int, int]:
    return height - height % 2, width - width % 2


def fit_max_side(height: int, width: int, max_side: int | None) -> tuple[int, int]:
    if max_side is None or max(height, width) <= max_side:
        return even_size(height, width)
    scale = max_side / max(height, width)
    return even_size(max(2, int(height * scale)), max(2, int(width * scale)))


def luma_to_bgr(luma: torch.Tensor) -> np.ndarray:
    gray = (luma.clamp(0, 1) * 255.0).to(torch.uint8)
    return gray.unsqueeze(-1).repeat(1, 1, 3).cpu().numpy()


class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def rewind(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *args) -> None:
        self.close()


def _drain_stream(stream: object, chunks: list[bytes]) -> None:
    read = getattr(stream, "read", None)
    if read is None:
        return
    while True:
        block = read(4096)
        if not block:
            break
        chunks.append(block)


class VideoWriter:
    """Encode an H.264 / yuv420p MP4 with ffmpeg libx264."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int) -> None:
        if width % 2 or height % 2:
            raise ValueError(f"H.264 yuv420p needs even size, got {width}x{height}")
        if width < 2 or height < 2:
            raise ValueError(f"Frame size too small: {width}x{height}")
        if fps <= 0:
            fps = 30.0

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg was not found on PATH. Install ffmpeg with libx264 to export an H.264 MP4."
            )

        self.path = str(path)
        self.width = width
        self.height = height
        self._closed = False
        self._stderr_chunks: list[bytes] = []
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:.6f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                "-preset",
                "fast",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                self.path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        if self._proc.stdin is None or self._proc.stderr is None:
            raise RuntimeError(f"Cannot create video writer: {self.path}")
        self._stdin = self._proc.stdin
        self._stderr_thread = threading.Thread(
            target=_drain_stream,
            args=(self._proc.stderr, self._stderr_chunks),
            daemon=True,
        )
        self._stderr_thread.start()

    def write(self, bgr: np.ndarray) -> None:
        if bgr.shape[0] != self.height or bgr.shape[1] != self.width or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"Expected {self.height}x{self.width}x3 frame, got {bgr.shape}")
        frame = np.ascontiguousarray(bgr, dtype=np.uint8)
        try:
            self._stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            self._finish(raise_on_error=False)
            raise RuntimeError(self._error_text("ffmpeg closed the encode pipe")) from exc

    def close(self) -> None:
        self._finish(raise_on_error=True)

    def _finish(self, raise_on_error: bool) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stdin.close()
        except BrokenPipeError:
            pass
        self._stderr_thread.join()
        code = self._proc.wait()
        if raise_on_error and code != 0:
            raise RuntimeError(self._error_text(f"ffmpeg exited with code {code}"))

    def _error_text(self, prefix: str) -> str:
        detail = b"".join(self._stderr_chunks).decode("utf-8", errors="replace").strip()
        if detail:
            return f"{prefix}: {self.path}\n{detail}"
        return f"{prefix}: {self.path}"

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class AsyncLumaWriter:
    """Double-buffer GPU luma -> CPU so encode can overlap with the next infer."""

    def __init__(self, path: str | Path, fps: float, width: int, height: int, device: torch.device) -> None:
        self.inner = VideoWriter(path, fps, width, height)
        self.device = device
        self.use_cuda = device.type == "cuda" and torch.cuda.is_available()
        self.idx = 0
        self.gpu_bufs: list[torch.Tensor] | None = None
        self.cpu_bufs: list[torch.Tensor] | None = None
        self.events: list[torch.cuda.Event | None] = [None, None]
        self.pending = [False, False]
        self.copy_stream: torch.cuda.Stream | None = None

    def _ensure_bufs(self, frame: torch.Tensor) -> None:
        if self.gpu_bufs is not None:
            return
        shape = tuple(frame.shape)
        self.gpu_bufs = [
            torch.empty(shape, dtype=torch.uint8, device=frame.device),
            torch.empty(shape, dtype=torch.uint8, device=frame.device),
        ]
        self.cpu_bufs = [
            torch.empty(shape, dtype=torch.uint8, device="cpu", pin_memory=True),
            torch.empty(shape, dtype=torch.uint8, device="cpu", pin_memory=True),
        ]
        self.copy_stream = torch.cuda.Stream(device=frame.device)

    def write_luma(self, luma: torch.Tensor) -> None:
        if not self.use_cuda:
            self.inner.write(luma_to_bgr(luma))
            return

        frame = (luma.clamp(0, 1) * 255.0).to(torch.uint8)
        frame = frame.unsqueeze(-1).expand(-1, -1, 3).contiguous()
        self._ensure_bufs(frame)
        assert self.gpu_bufs is not None and self.cpu_bufs is not None and self.copy_stream is not None

        i = self.idx
        if self.pending[i]:
            event = self.events[i]
            if event is not None:
                event.synchronize()
            self.inner.write(self.cpu_bufs[i].numpy())
            self.pending[i] = False

        self.gpu_bufs[i].copy_(frame)
        self.copy_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.copy_stream):
            self.cpu_bufs[i].copy_(self.gpu_bufs[i], non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.copy_stream)
            self.events[i] = event
        self.pending[i] = True
        self.idx ^= 1

    def close(self) -> None:
        if self.use_cuda and self.cpu_bufs is not None:
            for k in (self.idx, 1 - self.idx):
                if self.pending[k]:
                    event = self.events[k]
                    if event is not None:
                        event.synchronize()
                    self.inner.write(self.cpu_bufs[k].numpy())
                    self.pending[k] = False
        self.inner.close()

    def __enter__(self) -> AsyncLumaWriter:
        return self

    def __exit__(self, *args) -> None:
        self.close()
