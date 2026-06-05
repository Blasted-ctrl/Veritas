"""Classification metrics and ``metrics.json`` serialization.

``metrics.json`` is the single source of truth for reported performance. It is
always produced by evaluating a model on a held-out test split — numbers are
measured, never hand-written.
"""

from __future__ import annotations

import json
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
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = np.asarray(labels, dtype=int)
    p = np.asarray(probs, dtype=float)
    preds = (p >= threshold).astype(int)

    accuracy = float(accuracy_score(y, preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, preds, average="binary", zero_division=0, pos_label=1
    )

    # AUC is undefined with a single class present; fall back to NaN-safe 0.5.
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
    out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return out


__all__ = ["ClassificationMetrics", "compute_metrics", "write_metrics"]
