"""CNN checkpoints must keep the production ONNX pixel and capability contracts."""

from __future__ import annotations

import io
import json

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torchvision")
pytest.importorskip("onnxruntime")
pytest.importorskip("transformers")
pytestmark = pytest.mark.api


@pytest.mark.parametrize(
    "architecture,name",
    [("convnext_tiny", "Veritas ConvNeXt-Tiny"), ("efficientnet_b0", "Veritas EfficientNet-B0")],
)
def test_cnn_checkpoint_exports_and_matches_raw_pixel_logits(tmp_path, architecture, name):
    from veritas.api.inference import ImageDetector
    from veritas.export.onnx_export import export_image_from_dir
    from veritas.models.cnn_model import ConvNeXtDetector, EfficientNetDetector
    from veritas.models.image_model import make_image_transform

    old_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        torch.manual_seed(21)
        kind = ConvNeXtDetector if architecture == "convnext_tiny" else EfficientNetDetector
        model = kind(pretrained=False, image_size=32).eval()
        model.save_pretrained(tmp_path / "checkpoint")
        path = export_image_from_dir(tmp_path / "checkpoint", tmp_path / "cnn.onnx")
        detector = ImageDetector(path)
        assert detector.available
        pixels = (np.arange(17 * 39 * 3).reshape(17, 39, 3) * 37 % 256).astype(np.uint8)
        image = Image.fromarray(pixels)
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        with torch.no_grad():
            expected = model(pixel_values=make_image_transform(32)(image)[None]).logits.numpy()
        actual = detector._session.run(None, {"pixel_values": detector.preprocess(payload.getvalue())})[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
        metadata = json.loads(path.with_suffix(".onnx.json").read_text())
        assert metadata["image_size"] == 32
        assert metadata["id"] == name
        assert metadata["source_type"] == "custom"
        assert metadata["fake_index"] == 1
    finally:
        torch.set_num_threads(old_threads)


def test_vit_only_heatmaps_do_not_claim_cnn_support(tmp_path):
    from veritas.api.explainer import GradCamExplainer

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "veritas_convnext_tiny"}))
    explainer = GradCamExplainer(str(tmp_path))
    assert not explainer.available
    assert explainer.explain(b"not-an-image") is None
