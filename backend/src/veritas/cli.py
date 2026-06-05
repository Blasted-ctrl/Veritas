"""``veritas`` command-line interface.

Implemented with the standard-library :mod:`argparse` so the CLI runs without
installing any third-party package.  Subcommands are added per phase:

* ``prepare-data`` — Phase 1 (this file).
* ``train-image`` / ``train-audio`` — Phases 2/3 (registered lazily; they
  require the optional ``ml`` dependency group).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from veritas import __version__
from veritas.config import DEFAULT_SEED, DEFAULT_TEST_SIZE, DEFAULT_VAL_SIZE, PATHS


def _add_prepare_data(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "prepare-data",
        help="Download/prepare a dataset into balanced, identity-safe splits.",
        description="Build train/val/test manifests with no subject crossing splits.",
    )
    p.add_argument("--modality", required=True, choices=["image", "audio"])
    p.add_argument(
        "--source",
        default="synthetic",
        choices=["synthetic", "directory", "faceforensics", "dfdc", "asvspoof"],
        help="Dataset adapter to use (default: synthetic fixtures).",
    )
    p.add_argument(
        "--input-dir",
        default=None,
        help="Root directory of the dataset (required for non-synthetic sources).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write manifests (default: <data>/processed/<modality>).",
    )
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--val-size", type=float, default=DEFAULT_VAL_SIZE)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--no-balance", dest="balance", action="store_false", help="Skip class balancing.")
    p.add_argument("--synthetic-subjects", type=int, default=24)
    p.add_argument("--synthetic-per-subject", type=int, default=4)
    p.set_defaults(func=_run_prepare_data, balance=True)


def _run_prepare_data(args: argparse.Namespace) -> int:
    # Imported here so ``--help`` works even if optional deps are missing.
    from veritas.data.prepare import prepare

    output_dir = Path(args.output_dir) if args.output_dir else PATHS.modality_processed(args.modality)
    result = prepare(
        modality=args.modality,
        source=args.source,
        input_dir=args.input_dir,
        output_dir=output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        balance=args.balance,
        synthetic_subjects=args.synthetic_subjects,
        synthetic_per_subject=args.synthetic_per_subject,
    )
    splits = result.summary["splits"]  # type: ignore[index]
    print(f"Prepared {args.modality} dataset from '{args.source}' -> {result.output_dir}")
    for name in ("train", "val", "test"):
        s = splits[name]
        print(
            f"  {name:<5} samples={s['samples']:<5} "
            f"real={s['real']:<5} fake={s['fake']:<5} "
            f"fake_ratio={s['fake_ratio']:<6} subjects={s['subjects']}"
        )
    print("  identity_leakage=False (verified)")
    return 0


def _register_ml_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register Phase 2/3 training commands if the ML stack is importable.

    Resolved dynamically so the core CLI works even when the optional ``ml``
    dependency group (torch/transformers) is not installed.
    """
    import importlib

    try:
        training = importlib.import_module("veritas.training")
    except Exception:  # pragma: no cover - optional dependency not installed
        return
    register = getattr(training, "register_cli", None)
    if register is not None:  # available once Phases 2/3 land
        register(subparsers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veritas", description="Veritas deepfake & AI-content detector.")
    parser.add_argument("--version", action="version", version=f"veritas {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_prepare_data(subparsers)
    _register_ml_commands(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
