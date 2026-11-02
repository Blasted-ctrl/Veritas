"""Operating thresholds fitted only on a separate calibration split."""

from __future__ import annotations

import math
from bisect import bisect_left

from veritas.training.metrics import compute_metrics


def select_threshold(labels, scores, *, max_fpr: float = 0.05) -> dict:
    baseline = compute_metrics(labels, scores)
    if not baseline.n_positive or not baseline.n_negative:
        raise ValueError("Calibration requires both real and fake classes")
    if not math.isfinite(max_fpr) or not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be finite and in [0, 1]")
    real = sorted(float(score) for label, score in zip(labels, scores, strict=True) if label == 0)
    fake = sorted(float(score) for label, score in zip(labels, scores, strict=True) if label == 1)
    candidates = {0.0, 1.0, *real, *fake}
    candidates.update(math.nextafter(score, math.inf) for score in (*real, *fake) if score < 1)
    best = None
    for threshold in sorted(candidates):
        fp = len(real) - bisect_left(real, threshold)
        tp = len(fake) - bisect_left(fake, threshold)
        fpr = fp / len(real)
        balanced = (tp / len(fake) + 1 - fpr) / 2
        rank = (balanced, -fpr, -threshold)
        if fpr <= max_fpr and (best is None or rank > best[0]):
            best = (rank, threshold)
    if best is None:
        raise ValueError("No threshold in [0, 1] meets the calibration false-positive target")
    threshold = best[1]
    return {
        "threshold": threshold,
        "target_fpr": max_fpr,
        "calibration_real": len(real),
        "calibration_fake": len(fake),
        "method": "Max balanced accuracy within empirical FPR budget; ties: lower FPR, then threshold",
        "metrics": compute_metrics(labels, scores, threshold=threshold).to_dict(),
    }


def wilson_interval(events: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    if total < 1 or not 0 <= events <= total:
        raise ValueError("Invalid event/total counts")
    rate = events / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
