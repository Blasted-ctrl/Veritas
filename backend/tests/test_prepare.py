"""End-to-end tests for the prepare-data pipeline on synthetic fixtures."""

from __future__ import annotations

import json

import pytest

from veritas.data.manifest import FAKE, REAL, read_manifest
from veritas.data.prepare import balance_samples, prepare
from veritas.data.splitting import assert_no_identity_leakage


@pytest.mark.parametrize("modality", ["image", "audio"])
def test_prepare_produces_leakfree_balanced_splits(tmp_path, modality):
    result = prepare(
        modality=modality,
        source="synthetic",
        output_dir=tmp_path,
        synthetic_subjects=30,
        synthetic_per_subject=4,
        seed=1337,
    )

    # All three manifests written and non-empty.
    all_samples = []
    for name in ("train", "val", "test"):
        path = result.manifests[name]
        assert path.exists()
        rows = read_manifest(path)
        assert rows, f"{name} split is empty"
        all_samples.extend(rows)

    # The end-to-end output is leak-free.
    assert_no_identity_leakage(all_samples)

    # Media files actually exist on disk and decode as the right type.
    for s in all_samples[:5]:
        assert s.modality == modality
        with open(s.path, "rb") as fh:
            head = fh.read(4)
        if modality == "image":
            assert head.startswith(b"\x89PNG")
        else:
            assert head == b"RIFF"

    # summary.json records the leakage assertion and split stats.
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["identity_leakage"] is False
    assert summary["modality"] == modality
    assert summary["splits"]["total"]["samples"] == len(all_samples)


def test_prepare_is_deterministic(tmp_path):
    a = prepare(modality="image", output_dir=tmp_path / "a", seed=42, synthetic_subjects=20)
    b = prepare(modality="image", output_dir=tmp_path / "b", seed=42, synthetic_subjects=20)
    a_rows = {s.path.split("\\")[-1].split("/")[-1]: s.split for s in read_manifest(a.manifests["train"])}
    b_rows = {s.path.split("\\")[-1].split("/")[-1]: s.split for s in read_manifest(b.manifests["train"])}
    assert a_rows == b_rows


def test_balance_downsamples_majority_label():
    from veritas.data.manifest import Sample

    samples = [
        Sample(path=f"r{i}.png", label=REAL, subject_id=f"s{i}", modality="image", source="t")
        for i in range(10)
    ] + [
        Sample(path=f"f{i}.png", label=FAKE, subject_id=f"s{i}", modality="image", source="t")
        for i in range(4)
    ]
    balanced = balance_samples(samples, seed=1)
    reals = sum(1 for s in balanced if s.label == REAL)
    fakes = sum(1 for s in balanced if s.label == FAKE)
    assert reals == fakes == 4
