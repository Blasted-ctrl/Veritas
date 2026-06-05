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


# --------------------------------------------------------------------------- #
# train-image (Phase 2)
# --------------------------------------------------------------------------- #
def _add_train_image(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "train-image",
        help="Fine-tune a ViT real/manipulated image detector.",
        description="Fine-tune a Vision Transformer and write a fine-tuned model + metrics.json.",
    )
    p.add_argument("--data-dir", default=None, help="Dir with train/val/test.csv (default: processed/image).")
    p.add_argument(
        "--output-dir", default=None, help="Where to save model + metrics (default: models/image)."
    )
    p.add_argument("--model-name", default="google/vit-base-patch16-224")
    p.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Train a small randomly-initialised ViT (no download; for smoke runs).",
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
    output_dir = Path(args.output_dir) if args.output_dir else PATHS.models / "image"
    cfg = TrainImageConfig(
        data_dir=data_dir,
        output_dir=output_dir,
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
# train-audio (Phase 3 — registered now, handler lands in Phase 3)
# --------------------------------------------------------------------------- #
def _add_train_audio(subparsers: argparse._SubParsersAction) -> None:
    try:
        from veritas.training.train_audio import add_train_audio_parser
    except Exception:  # pragma: no cover - Phase 3 not yet present
        return
    add_train_audio_parser(subparsers)


__all__ = ["register_cli"]
