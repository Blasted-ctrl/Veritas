"""Tests for ViT Grad-CAM interpretability (tiny model, no network)."""

from __future__ import annotations

import base64
import io

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("PIL")

from veritas.data.synthetic import write_png  # noqa: E402
from veritas.explain.gradcam import ViTGradCAM, explain_image  # noqa: E402
from veritas.models.image_model import ImageModelConfig, build_vit, make_image_transform  # noqa: E402

pytestmark = pytest.mark.ml

SIZE = 32


def _tiny_vit():
    return build_vit(
        ImageModelConfig(
            pretrained=False,
            image_size=SIZE,
            patch_size=16,
            hidden_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=96,
        )
    )


def _png_bytes(tmp_path) -> bytes:
    p = tmp_path / "x.png"
    write_png(p, [[(i * 5 % 256, (j * 7) % 256, 0) for i in range(SIZE)] for j in range(SIZE)])
    return p.read_bytes()


def test_heatmap_shape_and_range(tmp_path):
    import torch
    from PIL import Image

    model = _tiny_vit()
    transform = make_image_transform(SIZE)
    with Image.open(io.BytesIO(_png_bytes(tmp_path))) as im:
        pixel_values = transform(im.convert("RGB")).unsqueeze(0)

    heat = ViTGradCAM(model).heatmap(pixel_values, class_idx=1)
    assert heat.shape == (SIZE, SIZE)
    assert float(heat.min()) >= 0.0 and float(heat.max()) <= 1.0
    assert isinstance(pixel_values, torch.Tensor)


def test_explain_image_returns_data_urls(tmp_path):
    result = explain_image(_tiny_vit(), _png_bytes(tmp_path), image_size=SIZE, class_idx=1)
    assert result["heatmap"].startswith("data:image/png;base64,")
    assert result["overlay"].startswith("data:image/png;base64,")
    assert result["grid_size"] == 2  # 32/16 = 2 -> 2x2 patch grid

    # The overlay decodes to a valid PNG of the input size.
    from PIL import Image

    raw = base64.b64decode(result["overlay"].split(",", 1)[1])
    with Image.open(io.BytesIO(raw)) as im:
        assert im.size == (SIZE, SIZE)
