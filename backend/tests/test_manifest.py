"""Tests for the Sample record and manifest CSV round-trip."""

from __future__ import annotations

import pytest

from veritas.data.manifest import (
    FAKE,
    REAL,
    Sample,
    label_counts,
    label_name,
    read_manifest,
    write_manifest,
)


def test_sample_validates_label():
    with pytest.raises(ValueError):
        Sample(path="x.png", label=2, subject_id="s1", modality="image", source="t")


def test_sample_validates_modality():
    with pytest.raises(ValueError):
        Sample(path="x.png", label=REAL, subject_id="s1", modality="video", source="t")


def test_sample_requires_subject():
    with pytest.raises(ValueError):
        Sample(path="x.png", label=REAL, subject_id="", modality="image", source="t")


def test_with_split_is_immutable_copy():
    s = Sample(path="x.png", label=REAL, subject_id="s1", modality="image", source="t")
    s2 = s.with_split("train")
    assert s.split == ""
    assert s2.split == "train"
    with pytest.raises(ValueError):
        s.with_split("holdout")


def test_manifest_round_trip(tmp_path):
    samples = [
        Sample(path="a.png", label=REAL, subject_id="s1", modality="image", source="t", split="train"),
        Sample(path="b.png", label=FAKE, subject_id="s2", modality="image", source="t", split="test"),
    ]
    out = write_manifest(samples, tmp_path / "m.csv")
    assert out.exists()
    loaded = read_manifest(out)
    assert loaded == samples


def test_label_helpers():
    assert label_name(REAL) == "real"
    assert label_name(FAKE) == "fake"
    counts = label_counts(
        [
            Sample(path="a", label=REAL, subject_id="s", modality="image", source="t"),
            Sample(path="b", label=FAKE, subject_id="s", modality="image", source="t"),
            Sample(path="c", label=FAKE, subject_id="s", modality="image", source="t"),
        ]
    )
    assert counts == {REAL: 1, FAKE: 2}
