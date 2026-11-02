"""Exports with a training audit must bind its provenance to completed checkpoint weights."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from veritas.export.onnx_export import export_image_from_dir  # noqa: E402
from veritas.models.image_model import ImageModelConfig, build_vit  # noqa: E402
from veritas.models.metadata import (  # noqa: E402
    load_image_metadata,
    metadata_path,
    torch_state_sha256,
)

pytestmark = pytest.mark.api

FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def _saved_model(model_dir):
    model = build_vit(
        ImageModelConfig(
            pretrained=False,
            image_size=32,
            patch_size=16,
            hidden_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=96,
        )
    )
    model.save_pretrained(model_dir)
    return model


def _write_audit(model_dir, fingerprint=FINGERPRINT_A):
    (model_dir / "data-audit.json").write_text(
        json.dumps({"fingerprint": fingerprint, "data_kind": "synthetic"}), encoding="utf-8"
    )


def _write_completion(model_dir, model):
    meta = {
        "dataset_fingerprint": FINGERPRINT_A,
        "torch_state_sha256": torch_state_sha256(model),
        "image_size": model.config.image_size,
    }
    (model_dir / "metrics.json").write_text(json.dumps({"meta": meta}), encoding="utf-8")
    return meta


def _assert_rejected_without_graph(model_dir, output_path):
    with pytest.raises(ValueError):
        export_image_from_dir(model_dir, output_path)
    assert not output_path.exists()
    assert not metadata_path(output_path).exists()


def test_completed_training_provenance_exports_the_bound_dataset(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model = _saved_model(model_dir)
    _write_audit(model_dir)
    completion = _write_completion(model_dir, model)

    output_path = export_image_from_dir(model_dir, tmp_path / "image.onnx")

    metadata = load_image_metadata(output_path)
    assert metadata["training_data_fingerprint"] == FINGERPRINT_A
    assert metadata["training_data_kind"] == "synthetic"
    assert metadata["torch_state_sha256"] == completion["torch_state_sha256"]
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"pixel_values": np.zeros((1, 3, 32, 32), dtype=np.float32)})[0]
    assert logits.shape == (1, 2)


def test_failed_rerun_audit_cannot_relabel_previously_completed_weights(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model = _saved_model(model_dir)
    _write_audit(model_dir)
    _write_completion(model_dir, model)
    # A later run finishes preflight on corpus B, then fails before saving weights.
    _write_audit(model_dir, FINGERPRINT_B)

    _assert_rejected_without_graph(model_dir, tmp_path / "image.onnx")


def test_weights_changed_after_completion_cannot_use_recorded_provenance(tmp_path):
    model_dir = tmp_path / "checkpoint"
    model = _saved_model(model_dir)
    _write_audit(model_dir)
    completion = _write_completion(model_dir, model)
    with torch.no_grad():
        model.classifier.bias.add_(1.0)
    model.save_pretrained(model_dir)
    assert torch_state_sha256(model) != completion["torch_state_sha256"]

    _assert_rejected_without_graph(model_dir, tmp_path / "image.onnx")


@pytest.mark.parametrize(
    "missing",
    ["metrics_file", "meta", "dataset_fingerprint", "torch_state_sha256", "image_size"],
)
def test_audit_without_complete_binding_cannot_claim_training_source(tmp_path, missing):
    model_dir = tmp_path / "checkpoint"
    model = _saved_model(model_dir)
    _write_audit(model_dir)
    completion = _write_completion(model_dir, model)
    metrics_path = model_dir / "metrics.json"
    if missing == "metrics_file":
        metrics_path.unlink()
    elif missing == "meta":
        metrics_path.write_text(json.dumps({}), encoding="utf-8")
    else:
        del completion[missing]
        metrics_path.write_text(json.dumps({"meta": completion}), encoding="utf-8")

    _assert_rejected_without_graph(model_dir, tmp_path / "image.onnx")
