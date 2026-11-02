"""Fine-tuning entry points (Phases 2 & 3).

``register_cli`` is discovered by :func:`veritas.cli._register_ml_commands` and
attaches the ``train-image`` / ``train-audio`` subcommands. Heavy imports
(torch/transformers) happen inside the command handlers, so importing this
package is cheap and never fails when the optional ``ml`` extra is absent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from veritas.config import DEFAULT_SEED, PATHS


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    _add_train_image(subparsers)
    _add_train_audio(subparsers)
    _add_evaluate_image(subparsers)


# --------------------------------------------------------------------------- #
# train-image (Phase 2)
# --------------------------------------------------------------------------- #
def _add_train_image(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train-image",
        help="Fine-tune a ConvNeXt CNN or ViT image detector.",
        description="Fine-tune a PyTorch image detector and evaluate a held-out test split.",
    )
    p.add_argument("--data-dir", default=None, help="Dir with train/val/test.csv (default: processed/image).")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to save model + metrics (default: models/cnn-image for CNN, models/image for ViT).",
    )
    p.add_argument("--architecture", choices=["convnext_tiny", "vit"], default="convnext_tiny")
    p.add_argument(
        "--model-name", default="google/vit-base-patch16-224", help="Checkpoint name for --architecture vit."
    )
    p.add_argument("--unfreeze-stages", type=int, default=1, help="Top CNN stages to fine-tune (0-4).")
    p.add_argument(
        "--heldout-generator",
        action="append",
        default=[],
        help="Generator family allowed only in test; repeat for several.",
    )
    p.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Use random initialization without downloading pretrained weights.",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--backbone-lr", type=float, default=5e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--unfreeze-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Cap samples per split (quick runs).")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto).")
    p.set_defaults(func=_run_train_image, pretrained=True)


def _run_train_image(args: argparse.Namespace) -> int:
    from veritas.training.train_image import TrainImageConfig, train_image

    data_dir = Path(args.data_dir) if args.data_dir else PATHS.modality_processed("image")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PATHS.models / ("cnn-image" if args.architecture == "convnext_tiny" else "image")
    )
    cfg = TrainImageConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        architecture=args.architecture,
        unfreeze_stages=args.unfreeze_stages,
        heldout_generators=tuple(args.heldout_generator),
        model_name=args.model_name,
        pretrained=args.pretrained,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        freeze_epochs=args.freeze_epochs,
        unfreeze_layers=args.unfreeze_layers,
        seed=args.seed,
        num_workers=args.num_workers,
        limit=args.limit,
        device=args.device,
    )
    train_image(cfg)
    return 0


# --------------------------------------------------------------------------- #
# train-audio (Phase 3)
# --------------------------------------------------------------------------- #
def _add_train_audio(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train-audio",
        help="Fine-tune a Wav2Vec2 real/synthetic voice detector.",
        description="Fine-tune Wav2Vec2 and write a fine-tuned model + metrics.json.",
    )
    p.add_argument("--data-dir", default=None, help="Dir with train/val/test.csv (default: processed/audio).")
    p.add_argument(
        "--output-dir", default=None, help="Where to save model + metrics (default: models/audio)."
    )
    p.add_argument("--model-name", default="facebook/wav2vec2-base")
    p.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train a small randomly-initialised Wav2Vec2 (no download; for smoke runs).",
    )
    p.add_argument("--max-seconds", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--backbone-lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--freeze-epochs", type=int, default=1)
    p.add_argument("--unfreeze-layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Cap samples per split (quick runs).")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto).")
    p.set_defaults(func=_run_train_audio, pretrained=True)


def _run_train_audio(args: argparse.Namespace) -> int:
    from veritas.training.train_audio import TrainAudioConfig, train_audio

    data_dir = Path(args.data_dir) if args.data_dir else PATHS.modality_processed("audio")
    output_dir = Path(args.output_dir) if args.output_dir else PATHS.models / "audio"
    cfg = TrainAudioConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        model_name=args.model_name,
        pretrained=args.pretrained,
        max_seconds=args.max_seconds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        freeze_epochs=args.freeze_epochs,
        unfreeze_layers=args.unfreeze_layers,
        seed=args.seed,
        num_workers=args.num_workers,
        limit=args.limit,
        device=args.device,
    )
    train_audio(cfg)
    return 0


def _add_evaluate_image(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("evaluate-image", help="Evaluate held-out images with a fixed ONNX model.")
    p.add_argument("--model", required=True, help="Exported ONNX classifier and its processing sidecar.")
    p.add_argument("--manifest", required=True, help="Image CSV with every split column set to test.")
    p.add_argument("--output", required=True, help="JSON report including every measured prediction.")
    p.add_argument(
        "--training-dir", help="Optional original train/val/test manifest directory for leakage checks."
    )
    p.add_argument("--threshold", type=float, default=0.5, help="Fixed before test evaluation; default 0.5.")
    p.set_defaults(func=_run_evaluate_image)


def _run_evaluate_image(args: argparse.Namespace) -> int:
    from veritas.training.evaluate_image import evaluate_image

    report = evaluate_image(
        args.model,
        args.manifest,
        output=args.output,
        training_dir=args.training_dir,
        threshold=args.threshold,
    )
    for variant, result in report["variants"].items():
        print(f"{variant}: {result['metrics']}")
    print(f"Saved measured evaluation -> {args.output}")
    return 0


__all__ = ["register_cli"]
