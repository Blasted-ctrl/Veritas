"""Pytest fixtures and path setup for the Veritas backend test suite."""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``src/`` importable when tests are invoked directly (belt-and-braces;
# pyproject also sets ``pythonpath = ["src"]``).
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402

from veritas.data.manifest import FAKE, REAL, Sample  # noqa: E402


def make_samples(
    *,
    subjects: int = 20,
    per_subject: int = 4,
    modality: str = "image",
    source: str = "unit",
) -> list[Sample]:
    """Build a balanced manifest where every subject has REAL and FAKE samples."""
    samples: list[Sample] = []
    for s in range(subjects):
        sid = f"subj_{s:03d}"
        for k in range(per_subject):
            label = REAL if k < per_subject // 2 else FAKE
            ext = "png" if modality == "image" else "wav"
            samples.append(
                Sample(
                    path=f"/data/{sid}_{k}.{ext}",
                    label=label,
                    subject_id=sid,
                    modality=modality,
                    source=source,
                )
            )
    return samples


@pytest.fixture
def balanced_samples() -> list[Sample]:
    return make_samples(subjects=40, per_subject=4)
