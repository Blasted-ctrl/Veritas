"""Measured classification metrics and JSON serialization.

Run metadata identifies whether results came from tuning or held-out evaluation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ClassificationMetrics:
    """Headline binary-classification metrics (positive class = FAKE)."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None
    n_samples: int
    n_positive: int
    n_negative: int
    confusion_matrix: dict[str, int]  # tp, fp, tn, fn
    threshold: float = 0.5
    balanced_accuracy: float | None = None
    average_precision: float | None = None
    false_positive_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    labels: Sequence[int],
    probs: Sequence[float],
    *,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute metrics from ground-truth labels and FAKE-class probabilities.

    Args:
        labels: Ground truth, 0 (REAL) / 1 (FAKE).
        probs: Predicted probability of the FAKE class in ``[0, 1]``.
        threshold: Decision threshold applied to ``probs``.
    """
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    if len(labels) != len(probs) or not len(labels):
        raise ValueError("Labels and scores must be nonempty and have the same length")
    if any(label not in (0, 1) for label in labels):
        raise ValueError("Labels must be binary (real=0, fake=1)")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("Scores must be finite and in [0, 1]")
    preds = (p >= threshold).astype(int)

    accuracy = float(accuracy_score(y, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0, pos_label=1
    )

    # Ranking metrics require positive and negative examples.
    if len(np.unique(y)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y, p))

    tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()

    return ClassificationMetrics(
        accuracy=round(accuracy, 4),
        precision=round(float(precision), 4),
        recall=round(float(recall), 4),
        f1=round(float(f1), 4),
        auc=round(auc, 4) if auc == auc else None,  # NaN -> None for JSON
        n_samples=int(len(y)),
        n_positive=int((y == 1).sum()),
        n_negative=int((y == 0).sum()),
        confusion_matrix={"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        threshold=threshold,
        balanced_accuracy=round(float((tp / (tp + fn) + tn / (tn + fp)) / 2), 4)
        if (tp + fn) and (tn + fp)
        else None,
        average_precision=round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) == 2 else None,
        false_positive_rate=round(float(fp / (tn + fp)), 4) if tn + fp else None,
    )


def write_metrics(
    metrics: ClassificationMetrics,
    path: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist ``metrics.json`` with optional run metadata under ``meta``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = metrics.to_dict()
    if extra:
        doc["meta"] = extra
    out.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    return out


__all__ = ["ClassificationMetrics", "compute_metrics", "write_metrics"]
