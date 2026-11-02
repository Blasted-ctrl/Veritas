"""Research reports preserve the operating-policy tradeoff and sealed data identity."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("sklearn")
Image = pytest.importorskip("PIL.Image")
from veritas.data.image_corpus import image_identity  # noqa: E402
from veritas.data.manifest import Sample, write_manifest  # noqa: E402
from veritas.models.metadata import file_sha256  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "diverse_evaluation", Path(__file__).parents[1] / "scripts" / "evaluate_diverse_cnn.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
pytestmark = pytest.mark.ml


def test_report_records_fewer_false_positives_at_calibrated_threshold():
    rows = [
        {"label": label, "fake_score": score, "source": "example", "generator_id": "g" if label else ""}
        for label, score in ((0, 0.1), (0, 0.6), (1, 0.7), (1, 0.9))
    ]
    fixed = evaluation.summarize(rows, 0.5)
    calibrated = evaluation.summarize(rows, 0.65)
    assert fixed["metrics"]["confusion_matrix"] == {"tp": 2, "fp": 1, "tn": 1, "fn": 0}
    assert calibrated["metrics"]["confusion_matrix"] == {"tp": 2, "fp": 0, "tn": 2, "fn": 0}
    assert calibrated["fpr_wilson_95"][1] > 0.5  # Two negatives do not establish a low population FPR.
    assert calibrated["by_generator"]["g"]["recall"] == 1


def test_changed_calibration_pixels_are_rejected_before_any_scoring(tmp_path):
    source = {"manifest_sha256": {}}
    for number, role in enumerate(("train", "val", "calibration", "id-test", "test")):
        image = tmp_path / (role + ".png")
        Image.new("RGB", (17, 19), (number * 30, 10, 20)).save(image)
        group = role + "-group"
        manifest = write_manifest(
            [Sample(str(image), 0, group, "image", "example", role if role in ("train", "val") else "test")],
            tmp_path / (role + ".csv"),
        )
        raw, pixels = image_identity(image)
        (tmp_path / (role + "-provenance.json")).write_text(
            json.dumps([{"downloaded_sha256": raw, "decoded_sha256": pixels, "group": group}])
        )
        source["manifest_sha256"][role] = file_sha256(manifest)
    evaluation.verify_roles(tmp_path, source)
    Image.new("RGB", (17, 19), (255, 50, 20)).save(tmp_path / "calibration.png")
    with pytest.raises(ValueError, match="image changed"):
        evaluation.verify_roles(tmp_path, source)


def test_processor_bundle_cannot_change_between_calibration_and_test(tmp_path, monkeypatch):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"fixed-graph")
    sidecar = Path(str(model) + ".json")
    sidecar.write_text('{"fake_index": 1}')
    identity = evaluation.bundle_identity(model)

    def changing(*args, **kwargs):
        sidecar.write_text('{"fake_index": 0}')
        return {"wrong_mapping": True}

    monkeypatch.setattr(evaluation, "evaluate_image", changing)
    output = tmp_path / "test-report.json"
    with pytest.raises(ValueError, match="bundle changed"):
        evaluation.evaluate_locked(model, tmp_path / "test.csv", tmp_path, identity, output)
    assert not output.exists()
