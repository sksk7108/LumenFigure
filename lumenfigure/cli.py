from __future__ import annotations

import argparse
from pathlib import Path

from lumenfigure.paths import DEFAULT_DEPTH_WEIGHTS, DEFAULT_RVM_WEIGHTS
from lumenfigure.pipeline import PipelineConfig, convert_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumenfigure",
        description="Extract people from video and render depth-shaded gray silhouettes.",
    )
    parser.add_argument("-i", "--input", required=True, help="Input video path")
    parser.add_argument("-o", "--output", required=True, help="Output mp4 path")
    parser.add_argument("--rvm-weights", type=Path, default=DEFAULT_RVM_WEIGHTS)
    parser.add_argument("--depth-weights", type=Path, default=DEFAULT_DEPTH_WEIGHTS)
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--depth-size", type=int, default=518)
    parser.add_argument(
        "--depth-stride",
        type=int,
        default=1,
        help="Kept for compatibility. Video Depth Anything streaming runs every frame, so this value is ignored.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1920,
        help="Downscale the long side before inference (1660 Ti default: 1920). Use 0 to keep original size.",
    )
    parser.add_argument(
        "--matte",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use RVM person matting. --no-matte shades the whole frame by depth.",
    )
    parser.add_argument(
        "--max-luma",
        "--highlight",
        dest="max_luma",
        type=float,
        default=1.0,
        help="Brightest near pixels (0-1). Alias: --highlight.",
    )
    parser.add_argument(
        "--min-luma",
        "--shadow",
        dest="min_luma",
        type=float,
        default=0.12,
        help="Darkest far-person floor (0-1). Alias: --shadow.",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.0,
        help="Depth separation: >1 stronger near/far contrast, <1 flatter. 1 is linear.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.85,
        help="Tone curve: <1 makes near parts pop, >1 compresses highlights.",
    )
    parser.add_argument(
        "--temporal-smooth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="EMA on depth and brightness stats to reduce flicker.",
    )
    parser.add_argument(
        "--blur-depth",
        action="store_true",
        help="Gaussian-blur the depth map for a flatter poster look.",
    )
    parser.add_argument("--blur-kernel", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    max_side = None if args.max_side == 0 else args.max_side
    cfg = PipelineConfig(
        input_path=Path(args.input),
        output_path=Path(args.output),
        rvm_weights=Path(args.rvm_weights),
        depth_weights=Path(args.depth_weights),
        device=args.device,
        fp16=args.fp16,
        depth_size=args.depth_size,
        depth_stride=args.depth_stride,
        max_side=max_side,
        use_matte=args.matte,
        gamma=args.gamma,
        contrast=args.contrast,
        min_luma=args.min_luma,
        max_luma=args.max_luma,
        temporal_smooth=args.temporal_smooth,
        blur_depth=args.blur_depth,
        blur_kernel=args.blur_kernel,
    )
    convert_video(cfg)
    return 0
