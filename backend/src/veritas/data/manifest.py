"""The :class:`Sample` record and manifest (CSV) serialization.

A manifest is just a list of :class:`Sample` rows.  We persist it as CSV so it
is human-inspectable, diff-friendly and trivially loadable by both the
stdlib-only data layer and (later) pandas/torch datasets.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

# Label convention used everywhere in the project:
#   REAL  -> authentic / bona-fide media
#   FAKE  -> manipulated / synthetic / spoofed media
REAL = 0
FAKE = 1

_LABEL_NAMES = {REAL: "real", FAKE: "fake"}

# Splits a manifest may be partitioned into.
SPLITS = ("train", "val", "test")

_FIELDS = ("path", "label", "subject_id", "modality", "source", "split")


def label_name(label: int) -> str:
    """Human-readable name for a label integer."""
    try:
        return _LABEL_NAMES[label]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(f"unknown label {label!r}; expected {REAL} or {FAKE}") from exc


@dataclass(frozen=True)
class Sample:
    """A single media item.

    Attributes:
        path: Location of the media file on disk.
        label: :data:`REAL` (0) or :data:`FAKE` (1).
        subject_id: Identity grouping key.  Every clip/frame/utterance of the
            *same* underlying person shares one ``subject_id`` so that the
            splitter can guarantee an identity never straddles two splits.
        modality: ``"image"`` or ``"audio"``.
        source: Originating dataset (e.g. ``"faceforensics"``, ``"synthetic"``).
        split: Assigned split (``"train"``/``"val"``/``"test"``) or ``""`` when
            the sample has not yet been split.
    """

    path: str
    label: int
    subject_id: str
    modality: str
    source: str
    split: str = ""

    def __post_init__(self) -> None:
        if self.label not in (REAL, FAKE):
            raise ValueError(f"label must be {REAL} or {FAKE}, got {self.label!r}")
        if self.modality not in ("image", "audio"):
            raise ValueError(f"modality must be 'image' or 'audio', got {self.modality!r}")
        if not str(self.subject_id):
            raise ValueError("subject_id must be a non-empty identifier")

    def with_split(self, split: str) -> Sample:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        return Sample(
            path=self.path,
            label=self.label,
            subject_id=self.subject_id,
            modality=self.modality,
            source=self.source,
            split=split,
        )


def write_manifest(samples: Sequence[Sample], path: str | Path) -> Path:
    """Write *samples* to ``path`` as CSV, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(asdict(sample))
    return out


def read_manifest(path: str | Path) -> list[Sample]:
    """Read a CSV manifest produced by :func:`write_manifest`."""
    rows: list[Sample] = []
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                Sample(
                    path=row["path"],
                    label=int(row["label"]),
                    subject_id=row["subject_id"],
                    modality=row["modality"],
                    source=row["source"],
                    split=row.get("split", "") or "",
                )
            )
    return rows


def label_counts(samples: Iterable[Sample]) -> dict[int, int]:
    """Count REAL/FAKE occurrences in *samples*."""
    counts = {REAL: 0, FAKE: 0}
    for sample in samples:
        counts[sample.label] += 1
    return counts


__all__ = [
    "Sample",
    "REAL",
    "FAKE",
    "SPLITS",
    "label_name",
    "label_counts",
    "read_manifest",
    "write_manifest",
]
