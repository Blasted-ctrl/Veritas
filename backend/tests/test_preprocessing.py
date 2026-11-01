"""Serving must feed the pixels described by the checkpoint's processor."""

from __future__ import annotations

import io
import json

import pytest

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

from veritas.api.inference import ImageDetector  # noqa: E402
from veritas.models.metadata import metadata_path  # noqa: E402

pytestmark = pytest.mark.api


def _pattern():
    values = (np.arange(17 * 39 * 3).reshape(17, 39, 3) * 37 % 256).astype(np.uint8)
    image = Image.fromarray(values)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return image, stream.getvalue()


def test_legacy_resize_matches_the_bilinear_training_transform(tmp_path):
    image, payload = _pattern()
    detector = ImageDetector(tmp_path / "model.onnx", image_size=32)
    actual = detector.preprocess(payload)
    resized = np.asarray(image.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    expected = (resized - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    np.testing.assert_allclose(actual, expected.transpose(2, 0, 1)[None], atol=1e-6)


def test_checkpoint_metadata_controls_normalization_and_input_size(tmp_path):
    image, payload = _pattern()
    model = tmp_path / "model.onnx"
    metadata_path(model).write_text(
        json.dumps(
            {
                "image_size": 24,
                "mean": [0.5] * 3,
                "std": [0.5] * 3,
                "resample": 2,
            }
        )
    )
    actual = ImageDetector(model).preprocess(payload)
    resized = np.asarray(image.resize((24, 24), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    expected = ((resized - 0.5) / 0.5).transpose(2, 0, 1)[None]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
