"""Held-out evaluation uses the real serving processor and rejects leakage."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("sklearn")
Image = pytest.importorskip("PIL.Image")
from veritas.data.manifest import Sample, write_manifest  # noqa: E402
from veritas.models.metadata import default_image_metadata, metadata_path  # noqa: E402
from veritas.training.evaluate_image import evaluate_image  # noqa: E402

pytestmark = pytest.mark.api


@pytest.fixture
def evaluation(tmp_path):
    # A real ONNX model: bright normalized pixels favor fake, dark favor real.
    from onnx import TensorProto, helper

    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("ReduceMean", ["pixel_values"], ["mean"], axes=[1, 2, 3], keepdims=0),
                helper.make_node("Neg", ["mean"], ["negative"]),
                helper.make_node("Unsqueeze", ["mean", "axes"], ["positive_column"]),
                helper.make_node("Unsqueeze", ["negative", "axes"], ["negative_column"]),
                helper.make_node("Concat", ["negative_column", "positive_column"], ["logits"], axis=1),
            ],
            "brightness",
            [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [None, 3, 32, 32])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 2])],
            [helper.make_tensor("axes", TensorProto.INT64, [1], [1])],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=9,
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)
    metadata_path(path).write_text(json.dumps(default_image_metadata(32)))
    rows = []
    for label, value in ((0, 20), (1, 235)):
        image = tmp_path / f"sample-{label}.png"
        Image.new("RGB", (19, 27), (value, value + 2, value + 4)).save(image)
        rows.append(
            Sample(
                image.name,
                label,
                f"group-{label}",
                "image",
                "test-source",
                "test",
                generator_id="test-generator" if label else "",
            )
        )
    manifest = write_manifest(rows, tmp_path / "test.csv")
    return path, manifest, rows


def test_real_onnx_predictions_variants_and_single_class_groups(evaluation, tmp_path):
    model, manifest, _ = evaluation
    output = tmp_path / "report.json"
    report = evaluate_image(model, manifest, output=output)
    assert json.loads(output.read_text()) == report
    assert len(report["model"]["artifact_sha256"]) == 64
    assert report["separation"]["training_reference_checked"] is False
    for variant in ("original", "jpeg75", "downscale50"):
        result = report["variants"][variant]
        assert result["metrics"]["confusion_matrix"] == {"tp": 1, "fp": 0, "tn": 1, "fn": 0}
        assert result["metrics"]["balanced_accuracy"] == 1
        assert len(result["predictions"]) == 2
        assert result["by_generator"]["test-generator"]["n_samples"] == 1
        assert result["by_generator"]["test-generator"]["auc"] is None
        assert result["by_source"]["test-source"]["n_samples"] == 2


def test_training_rows_cannot_be_reported_as_held_out(evaluation):
    model, manifest, rows = evaluation
    write_manifest([replace(rows[0], split="train"), rows[1]], manifest)
    with pytest.raises(ValueError, match="test"):
        evaluate_image(model, manifest)


def test_reencoded_duplicate_cannot_inflate_evaluation(evaluation):
    model, manifest, rows = evaluation
    clone = manifest.parent / "reencoded.png"
    info = __import__("PIL.PngImagePlugin", fromlist=["PngInfo"]).PngInfo()
    info.add_text("caption", "same pixels")
    with Image.open(manifest.parent / rows[0].path) as image:
        image.save(clone, pnginfo=info)
    write_manifest(rows + [replace(rows[0], path=clone.name, subject_id="another-id")], manifest)
    with pytest.raises(ValueError, match="uplicate"):
        evaluate_image(model, manifest)


def test_threshold_is_checked_before_report_is_written(evaluation, tmp_path):
    model, manifest, _ = evaluation
    output = tmp_path / "report.json"
    with pytest.raises(ValueError, match="hreshold"):
        evaluate_image(model, manifest, output=output, threshold=float("nan"))
    assert not output.exists()


def test_reference_corpus_prevents_group_or_pixel_leakage(evaluation, tmp_path):
    model, manifest, rows = evaluation
    reference = tmp_path / "training-reference"
    reference.mkdir()
    for number, split in enumerate(("train", "val", "test")):
        split_rows = []
        for label in (0, 1):
            path = reference / f"{split}-{label}.png"
            Image.new("RGB", (13, 17), (50 + number * 20, label * 100, 20)).save(path)
            split_rows.append(Sample(str(path), label, f"{split}-{label}", "image", "reference", split))
        write_manifest(split_rows, reference / (split + ".csv"))
    valid = evaluate_image(model, manifest, training_dir=reference)
    assert valid["separation"]["training_reference_checked"] is True
    write_manifest([replace(rows[0], subject_id="train-0"), rows[1]], manifest)
    with pytest.raises(ValueError, match="group"):
        evaluate_image(model, manifest, training_dir=reference)
    # Disguising the group does not bypass identical decoded pixels.
    with Image.open(reference / "train-0.png") as image:
        image.save(manifest.parent / rows[0].path)
    write_manifest(rows, manifest)
    with pytest.raises(ValueError, match="content"):
        evaluate_image(model, manifest, training_dir=reference)
