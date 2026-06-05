"""End-to-end data preparation: load -> balance -> split -> persist.

Produces, under ``<output_dir>``:

* ``train.csv`` / ``val.csv`` / ``test.csv`` — per-split manifests.
* ``summary.json`` — split sizes, label balance, subject counts, and an
  explicit ``identity_leakage: false`` assertion record.

The pipeline always re-verifies leak-freedom before writing, so a corrupt or
mis-configured run fails loudly rather than silently producing a leaky split.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from veritas.data import datasets, synthetic
from veritas.data.manifest import FAKE, REAL, SPLITS, Sample, write_manifest
from veritas.data.splitting import (
    assert_no_identity_leakage,
    split_manifest,
    summarize_splits,
)


@dataclass
class PrepareResult:
    output_dir: Path
    manifests: dict[str, Path]
    summary: dict[str, Any]


def balance_samples(samples: Sequence[Sample], *, seed: int = 1337) -> list[Sample]:
    """Down-sample the majority label so REAL and FAKE counts match.

    Balancing happens at the sample level and is independent of splitting, so it
    never introduces identity leakage.  Deterministic given ``seed``.
    """
    by_label: dict[int, list[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)
    if not by_label[REAL] or not by_label[FAKE]:
        return list(samples)

    keep = min(len(by_label[REAL]), len(by_label[FAKE]))
    rng = random.Random(seed)
    balanced: list[Sample] = []
    for label in (REAL, FAKE):
        pool = sorted(by_label[label], key=lambda s: s.path)
        rng.shuffle(pool)
        balanced.extend(pool[:keep])
    balanced.sort(key=lambda s: (s.subject_id, s.path))
    return balanced


def load_samples(
    *,
    modality: str,
    source: str,
    input_dir: str | Path | None,
    output_dir: Path,
    synthetic_subjects: int = 24,
    synthetic_per_subject: int = 4,
    seed: int = 1337,
) -> list[Sample]:
    """Load a manifest from ``source`` (real dataset or synthetic fixtures)."""
    if source == "synthetic":
        raw_dir = output_dir / "raw"
        return synthetic.generate_dataset(
            raw_dir,
            modality,
            subjects=synthetic_subjects,
            per_subject=synthetic_per_subject,
            seed=seed,
        )
    if input_dir is None:
        raise ValueError(f"source {source!r} requires --input-dir")
    return datasets.load_source(source, modality, input_dir)


def prepare(
    *,
    modality: str,
    source: str = "synthetic",
    input_dir: str | Path | None = None,
    output_dir: str | Path,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 1337,
    balance: bool = True,
    synthetic_subjects: int = 24,
    synthetic_per_subject: int = 4,
) -> PrepareResult:
    """Run the full preparation pipeline and write split manifests to disk."""
    if modality not in ("image", "audio"):
        raise ValueError(f"modality must be 'image' or 'audio', got {modality!r}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    samples = load_samples(
        modality=modality,
        source=source,
        input_dir=input_dir,
        output_dir=out,
        synthetic_subjects=synthetic_subjects,
        synthetic_per_subject=synthetic_per_subject,
        seed=seed,
    )
    if balance:
        samples = balance_samples(samples, seed=seed)

    split = split_manifest(samples, test_size=test_size, val_size=val_size, seed=seed)

    # Hard gate: never write a leaky split.
    assert_no_identity_leakage(split)

    manifests: dict[str, Path] = {}
    for name in SPLITS:
        members = [s for s in split if s.split == name]
        manifests[name] = write_manifest(members, out / f"{name}.csv")

    summary = summarize_splits(split)
    summary_doc = {
        "modality": modality,
        "source": source,
        "seed": seed,
        "balanced": balance,
        "test_size": test_size,
        "val_size": val_size,
        "identity_leakage": False,  # asserted above
        "splits": summary,
    }
    (out / "summary.json").write_text(json.dumps(summary_doc, indent=2), encoding="utf-8")

    return PrepareResult(output_dir=out, manifests=manifests, summary=summary_doc)


__all__ = ["prepare", "balance_samples", "load_samples", "PrepareResult"]
