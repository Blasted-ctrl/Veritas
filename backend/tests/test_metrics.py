"""Tests for the metrics module (requires the ml extra: numpy + scikit-learn)."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("numpy")

from veritas.training.metrics import compute_metrics, write_metrics  # noqa: E402

pytestmark = pytest.mark.ml


def test_perfect_predictions():
    labels = [0, 0, 1, 1]
    probs = [0.1, 0.2, 0.8, 0.9]
    m = compute_metrics(labels, probs)
    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.auc == 1.0
    assert m.confusion_matrix == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
    assert m.n_samples == 4


def test_all_wrong_predictions():
    labels = [0, 0, 1, 1]
    probs = [0.9, 0.8, 0.2, 0.1]
    m = compute_metrics(labels, probs)
    assert m.accuracy == 0.0
    assert m.auc == 0.0


def test_single_class_auc_is_none():
    m = compute_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert m.auc is None  # AUC undefined with one class


def test_write_metrics_roundtrip(tmp_path):
    m = compute_metrics([0, 1], [0.2, 0.8])
    out = write_metrics(m, tmp_path / "metrics.json", extra={"modality": "image", "seed": 1337})
    doc = json.loads(out.read_text())
    assert doc["accuracy"] == 1.0
    assert doc["meta"]["modality"] == "image"
    assert doc["meta"]["seed"] == 1337
