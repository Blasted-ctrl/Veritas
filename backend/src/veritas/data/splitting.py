"""Identity-safe, balanced train/val/test splitting.

The single hard requirement of this module: **a subject identity must never
appear in more than one split.**  If a person's authentic frames are in the
training set and their deepfaked frames are in the test set, a model can cheat
by memorising identity rather than learning manipulation artefacts, and the
reported test metrics become meaningless.

We guarantee leak-freedom *structurally*: samples are grouped by
``subject_id`` and every group is assigned, in its entirety, to exactly one
split.  Within that hard constraint we use a deterministic greedy algorithm
that simultaneously targets (a) the requested split proportions and (b) a
balanced REAL/FAKE ratio inside every split.

No third-party dependencies — this runs on a bare interpreter so the leakage
guarantee is always testable in CI.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from veritas.data.manifest import FAKE, REAL, SPLITS, Sample


class LeakageError(AssertionError):
    """Raised when a subject identity is found in more than one split."""


@dataclass
class _Group:
    """A subject identity and the per-label counts of its samples."""

    subject_id: str
    counts: dict[int, int]

    @property
    def size(self) -> int:
        return sum(self.counts.values())


@dataclass
class _Bin:
    """A split being filled by the greedy packer."""

    name: str
    target_total: float
    target_label: dict[int, float]
    total: int = 0
    label_total: dict[int, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.label_total = {REAL: 0, FAKE: 0}

    def deficit_score(self, group: _Group) -> tuple[float, float]:
        """How well *group* fits this bin's remaining need.

        Returns a 2-tuple sort key (higher is a better fit):
          1. label-aware deficit: rewards placing a group's REAL/FAKE mass into
             the bin that is furthest below its per-label target;
          2. overall size deficit: tie-breaker that keeps bins proportional.
        """
        label_fit = 0.0
        for label, count in group.counts.items():
            deficit = self.target_label[label] - self.label_total[label]
            label_fit += deficit * count
        size_fit = self.target_total - self.total
        return (label_fit, size_fit)

    def add(self, group: _Group) -> None:
        self.total += group.size
        for label, count in group.counts.items():
            self.label_total[label] += count


def _build_groups(samples: Sequence[Sample]) -> list[_Group]:
    counts: dict[str, dict[int, int]] = defaultdict(lambda: {REAL: 0, FAKE: 0})
    for sample in samples:
        counts[sample.subject_id][sample.label] += 1
    return [_Group(subject_id=sid, counts=dict(c)) for sid, c in counts.items()]


def split_manifest(
    samples: Sequence[Sample],
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 1337,
) -> list[Sample]:
    """Assign every sample a ``split`` such that no subject crosses splits.

    Args:
        samples: The manifest to split (``split`` field is ignored/overwritten).
        test_size: Fraction of *samples* targeted for the test split.
        val_size: Fraction targeted for the validation split.
        seed: RNG seed for deterministic, reproducible assignment.

    Returns:
        A new list of :class:`Sample` with ``split`` populated.

    Raises:
        ValueError: if the requested fractions are invalid, or if there are
            fewer distinct subjects than splits (which makes a leak-free split
            of all three partitions impossible).
    """
    if not samples:
        return []
    if not (0.0 < test_size < 1.0) or not (0.0 < val_size < 1.0):
        raise ValueError("test_size and val_size must each be in (0, 1)")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0 (train needs a share)")

    groups = _build_groups(samples)
    if len(groups) < len(SPLITS):
        raise ValueError(
            f"need at least {len(SPLITS)} distinct subject identities to build "
            f"leak-free train/val/test splits, got {len(groups)}"
        )

    total = sum(g.size for g in groups)
    label_totals = {REAL: 0, FAKE: 0}
    for g in groups:
        for label, count in g.counts.items():
            label_totals[label] += count

    train_size = 1.0 - test_size - val_size
    fractions = {"train": train_size, "val": val_size, "test": test_size}
    bins = {
        name: _Bin(
            name=name,
            target_total=frac * total,
            target_label={label: frac * label_totals[label] for label in (REAL, FAKE)},
        )
        for name, frac in fractions.items()
    }

    # Deterministic ordering: shuffle for unbiased tie-breaking, then place the
    # largest groups first (classic greedy bin-packing reduces final imbalance).
    rng = random.Random(seed)
    ordered = sorted(groups, key=lambda g: g.subject_id)
    rng.shuffle(ordered)
    ordered.sort(key=lambda g: g.size, reverse=True)

    assignment: dict[str, str] = {}
    # Seed each bin with one distinct group first so every split is non-empty
    # even on small datasets, prioritising the largest target shares.
    seed_order = sorted(bins.values(), key=lambda b: b.target_total, reverse=True)
    for bin_, group in zip(seed_order, ordered, strict=False):
        bin_.add(group)
        assignment[group.subject_id] = bin_.name
    remaining = ordered[len(seed_order) :]

    for group in remaining:
        best = max(bins.values(), key=lambda b: b.deficit_score(group))
        best.add(group)
        assignment[group.subject_id] = best.name

    return [s.with_split(assignment[s.subject_id]) for s in samples]


def assert_no_identity_leakage(samples: Sequence[Sample]) -> None:
    """Raise :class:`LeakageError` if any subject appears in multiple splits."""
    subject_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        if sample.split:
            subject_splits[sample.subject_id].add(sample.split)

    leaked = {sid: sorted(sp) for sid, sp in subject_splits.items() if len(sp) > 1}
    if leaked:
        preview = ", ".join(f"{sid} -> {sp}" for sid, sp in list(leaked.items())[:5])
        raise LeakageError(
            f"{len(leaked)} subject identit{'y' if len(leaked) == 1 else 'ies'} "
            f"cross split boundaries (identity leakage): {preview}"
        )


def summarize_splits(samples: Sequence[Sample]) -> dict[str, dict[str, object]]:
    """Per-split counts, label balance and subject totals for reporting."""
    summary: dict[str, dict[str, object]] = {}
    subjects: dict[str, set[str]] = defaultdict(set)
    for split in SPLITS:
        members = [s for s in samples if s.split == split]
        real = sum(1 for s in members if s.label == REAL)
        fake = sum(1 for s in members if s.label == FAKE)
        for s in members:
            subjects[split].add(s.subject_id)
        total = real + fake
        summary[split] = {
            "samples": total,
            "real": real,
            "fake": fake,
            "fake_ratio": round(fake / total, 4) if total else 0.0,
            "subjects": len(subjects[split]),
        }
    summary["total"] = {
        "samples": len(samples),
        "subjects": len({s.subject_id for s in samples}),
    }
    return summary


__all__ = [
    "split_manifest",
    "assert_no_identity_leakage",
    "summarize_splits",
    "LeakageError",
]
