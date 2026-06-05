"""Dataset loading, manifest construction and identity-safe splitting.

This subpackage is standard-library only.  A *manifest* is the central
abstraction: an ordered collection of :class:`~veritas.data.manifest.Sample`
records, each pairing a media file with its authenticity label and the subject
identity used to prevent train/test leakage.
"""

from __future__ import annotations

from veritas.data.manifest import (
    FAKE,
    REAL,
    Sample,
    label_name,
    read_manifest,
    write_manifest,
)
from veritas.data.splitting import (
    LeakageError,
    assert_no_identity_leakage,
    split_manifest,
    summarize_splits,
)

__all__ = [
    "Sample",
    "REAL",
    "FAKE",
    "label_name",
    "read_manifest",
    "write_manifest",
    "split_manifest",
    "assert_no_identity_leakage",
    "summarize_splits",
    "LeakageError",
]
