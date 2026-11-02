"""Low-false-positive operating points are fitted without using test predictions."""

import pytest

pytest.importorskip("sklearn")
from veritas.training.calibration import select_threshold, wilson_interval  # noqa: E402

pytestmark = pytest.mark.ml


def test_tied_real_score_is_excluded_without_discarding_more_fakes():
    report = select_threshold([0] * 10 + [1] * 10, [0.1] * 9 + [0.8] + [0.9] * 9 + [0.8], max_fpr=0.05)
    assert 0.8 < report["threshold"] < 0.9
    assert report["metrics"]["confusion_matrix"] == {"tp": 9, "fp": 0, "tn": 10, "fn": 1}
    assert report["metrics"]["balanced_accuracy"] == 0.95


def test_all_real_predictions_do_not_win_when_useful_detection_meets_fpr_budget():
    report = select_threshold([0, 0, 1, 1], [0.1, 0.4, 0.5, 0.8], max_fpr=0)
    assert report["metrics"]["accuracy"] == 1
    assert 0.4 < report["threshold"] <= 0.5


def test_no_legal_threshold_is_reported_when_scores_saturate():
    with pytest.raises(ValueError, match="threshold"):
        select_threshold([0, 0, 1, 1], [1.0, 1.0, 1.0, 1.0], max_fpr=0.05)


def test_single_class_data_cannot_calibrate_a_detector():
    with pytest.raises(ValueError, match="both"):
        select_threshold([0, 0], [0.2, 0.4])


def test_zero_observed_false_positives_does_not_imply_zero_risk():
    lower, upper = wilson_interval(0, 32)
    assert lower == pytest.approx(0)
    assert upper == pytest.approx(0.10718, abs=0.0001)
