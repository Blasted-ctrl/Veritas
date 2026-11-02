"""The CPU-feasible CNN trains convolution weights and reloads exact predictions."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from veritas.models.cnn_model import EfficientNetDetector  # noqa:E402
from veritas.models.image_model import load_image_classifier  # noqa:E402

pytestmark = pytest.mark.ml


def test_efficientnet_cached_prefix_matches_full_forward_and_updates_convolutions(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(71)
    model = EfficientNetDetector(pretrained=False, image_size=32).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.features[7:].parameters():
        parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        cached = model.features[:7](pixels)
        expected = model(pixel_values=pixels).logits
        actual = model.classifier(model.avgpool(model.features[7:](cached)))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    prefix = model.features[0][0].weight.detach().clone()
    upper = next(model.features[7].parameters()).detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    cached = torch.randn_like(cached)
    torch.nn.functional.cross_entropy(
        model.classifier(model.avgpool(model.features[7:](cached))), torch.tensor([0, 1])
    ).backward()
    assert next(model.features[7].parameters()).grad.abs().max() > 1e-8
    optimizer.step()
    assert torch.equal(model.features[0][0].weight, prefix)
    assert not torch.equal(next(model.features[7].parameters()), upper)
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "veritas_efficientnet_b0"
    loaded = load_image_classifier(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(pixel_values=pixels).logits, model(pixel_values=pixels).logits, rtol=0, atol=0
        )
