"""ONNX export of trained models (Phase 4). Requires the ``ml`` extra.

Exposes ``register_cli`` so the top-level CLI gains a ``export`` subcommand when
the ML stack is installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from veritas.config import PATHS


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "export",
        help="Export a fine-tuned model to ONNX.",
        description="Export a saved ViT/Wav2Vec2 model directory to an ONNX graph for serving.",
    )
    p.add_argument("--modality", required=True, choices=["image", "audio"])
    p.add_argument("--model-dir", required=True, help="Directory written by train-image/train-audio.")
    p.add_argument(
        "--output", default=None, help="ONNX output path (default: <models>/<modality>/model.onnx)."
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--sample-length", type=int, default=16000)
    p.set_defaults(func=_run_export)


def _run_export(args: argparse.Namespace) -> int:
    from veritas.export.onnx_export import export_audio_from_dir, export_image_from_dir

    output = Path(args.output) if args.output else PATHS.models / args.modality / "model.onnx"
    if args.modality == "image":
        path = export_image_from_dir(args.model_dir, output, image_size=args.image_size)
    else:
        path = export_audio_from_dir(args.model_dir, output, sample_length=args.sample_length)
    print(f"Exported {args.modality} model -> {path}")
    return 0


__all__ = ["register_cli"]
