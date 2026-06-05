"""Tests for ONNX export (tiny models, no network)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")

from veritas.export.onnx_export import export_audio_onnx, export_image_onnx  # noqa: E402
from veritas.models.audio_model import AudioModelConfig, build_wav2vec2  # noqa: E402
from veritas.models.image_model import ImageModelConfig, build_vit  # noqa: E402

pytestmark = pytest.mark.api


def _tiny_vit():
    return build_vit(
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


def _tiny_w2v():
    return build_wav2vec2(
        AudioModelConfig(pretrained=False, hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
    )


def test_export_image_onnx_runs_in_onnxruntime(tmp_path):
    import onnxruntime as ort

    path = export_image_onnx(_tiny_vit(), tmp_path / "image.onnx", image_size=32)
    assert path.exists()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"pixel_values": np.zeros((1, 3, 32, 32), dtype=np.float32)})[0]
    assert out.shape == (1, 2)


def test_export_audio_onnx_runs_in_onnxruntime(tmp_path):
    import onnxruntime as ort

    path = export_audio_onnx(_tiny_w2v(), tmp_path / "audio.onnx", sample_length=4000)
    assert path.exists()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    # Dynamic time axis: a different length than the export dummy must work.
    out = sess.run(["logits"], {"input_values": np.zeros((1, 3200), dtype=np.float32)})[0]
    assert out.shape == (1, 2)
