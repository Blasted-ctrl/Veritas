"""Model provenance and installation contracts; no network or ML required."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

from veritas.models.metadata import load_image_metadata, metadata_path
from veritas.models.pretrained import download_model


def test_legacy_custom_metadata_matches_training_resize(tmp_path):
    info = load_image_metadata(tmp_path / "model.onnx", image_size=32)
    assert info["source_type"] == "custom"
    assert info["image_size"] == 32
    assert info["resample"] == 2
    assert info["fake_index"] == 1


def test_manifest_preserves_processor_and_reversed_labels(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model fixture")
    metadata_path(model).write_text(
        json.dumps(
            {
                "format_version": 1,
                "image_size": 64,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "fake_index": 0,
                "artifact_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            }
        )
    )
    info = load_image_metadata(model)
    assert info["fake_index"] == 0
    assert info["mean"] == [0.5, 0.5, 0.5]
    assert info["image_size"] == 64


@pytest.mark.parametrize("field,value", [("std", [0, 1, 1]), ("mean", [0.5]), ("fake_index", 3)])
def test_invalid_processor_is_rejected(tmp_path, field, value):
    model = tmp_path / "model.onnx"
    metadata_path(model).write_text(json.dumps({field: value}))
    with pytest.raises(ValueError):
        load_image_metadata(model)


def test_manifest_checksum_must_match_artifact(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"changed model")
    metadata_path(model).write_text(json.dumps({"artifact_sha256": "0" * 64}))
    with pytest.raises(ValueError, match="checksum"):
        load_image_metadata(model)


def test_failed_download_preserves_existing_model(tmp_path, monkeypatch):
    import veritas.models.pretrained as pretrained

    model = tmp_path / "model.onnx"
    model.write_bytes(b"previous working model")
    monkeypatch.setattr(pretrained, "urlopen", lambda *args, **kwargs: io.BytesIO(b"wrong weights"))
    with pytest.raises(ValueError):
        download_model(model, overwrite=True)
    assert model.read_bytes() == b"previous working model"
    assert not list(tmp_path.glob("*.partial"))
