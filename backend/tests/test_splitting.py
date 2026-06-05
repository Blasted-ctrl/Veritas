"""Tests for identity-safe, balanced splitting — the Phase 1 acceptance gate."""

from __future__ import annotations

import pytest

from conftest import make_samples
from veritas.data.splitting import (
    LeakageError,
    assert_no_identity_leakage,
    split_manifest,
    summarize_splits,
)


def _subjects_by_split(samples):
    out: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for s in samples:
        out[s.split].add(s.subject_id)
    return out


# --- The headline requirement -------------------------------------------------
def test_no_subject_identity_crosses_splits():
    """A subject identity must appear in at most one split (no leakage)."""
    samples = make_samples(subjects=60, per_subject=6)
    split = split_manifest(samples, test_size=0.2, val_size=0.2, seed=1337)

    by_split = _subjects_by_split(split)
    train, val, test = by_split["train"], by_split["val"], by_split["test"]

    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    # Every subject landed somewhere.
    assert len(train | val | test) == 60
    # And the dedicated guard agrees.
    assert_no_identity_leakage(split)


def test_assert_no_identity_leakage_detects_a_planted_leak():
    samples = make_samples(subjects=10, per_subject=4)
    split = split_manifest(samples, seed=7)
    # Force one subject to straddle two splits.
    leaked_subject = split[0].subject_id
    tampered = []
    flipped = False
    for s in split:
        if s.subject_id == leaked_subject and not flipped:
            other = "test" if s.split != "test" else "train"
            tampered.append(s.with_split(other))
            flipped = True
        else:
            tampered.append(s)
    with pytest.raises(LeakageError):
        assert_no_identity_leakage(tampered)


# --- Balance ------------------------------------------------------------------
def test_each_split_is_class_balanced():
    samples = make_samples(subjects=80, per_subject=4)  # globally 50/50
    split = split_manifest(samples, test_size=0.15, val_size=0.15, seed=1337)
    summary = summarize_splits(split)
    for name in ("train", "val", "test"):
        s = summary[name]
        assert s["samples"] > 0
        # Within 0.5 +/- 0.15 — group constraints prevent perfect balance.
        assert 0.35 <= s["fake_ratio"] <= 0.65, (name, s)


# --- Proportions --------------------------------------------------------------
def test_split_sizes_track_requested_fractions():
    samples = make_samples(subjects=100, per_subject=4)
    split = split_manifest(samples, test_size=0.2, val_size=0.2, seed=1337)
    summary = summarize_splits(split)
    total = summary["total"]["samples"]
    assert summary["train"]["samples"] / total == pytest.approx(0.6, abs=0.12)
    assert summary["val"]["samples"] / total == pytest.approx(0.2, abs=0.12)
    assert summary["test"]["samples"] / total == pytest.approx(0.2, abs=0.12)


# --- Determinism --------------------------------------------------------------
def test_same_seed_is_reproducible():
    samples = make_samples(subjects=50, per_subject=4)
    a = split_manifest(samples, seed=2024)
    b = split_manifest(samples, seed=2024)
    assert [s.split for s in a] == [s.split for s in b]


def test_different_seeds_can_differ():
    samples = make_samples(subjects=50, per_subject=4)
    a = [s.split for s in split_manifest(samples, seed=1)]
    b = [s.split for s in split_manifest(samples, seed=999)]
    assert a != b  # extremely unlikely to coincide for 50 subjects


# --- Guard rails --------------------------------------------------------------
def test_rejects_too_few_subjects():
    samples = make_samples(subjects=2, per_subject=4)
    with pytest.raises(ValueError):
        split_manifest(samples)


def test_rejects_impossible_fractions():
    samples = make_samples(subjects=10, per_subject=4)
    with pytest.raises(ValueError):
        split_manifest(samples, test_size=0.6, val_size=0.6)


def test_empty_manifest_returns_empty():
    assert split_manifest([]) == []
