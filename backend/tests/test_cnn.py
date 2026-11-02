"""Real CNN gradients and checkpoint integrity; no downloaded weights."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("PIL")
pytestmark = pytest.mark.ml


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def test_cnn_gradients_update_head_then_only_top_stage():
    from veritas.models.cnn_model import ConvNeXtDetector, unfreeze_top_stages
    from veritas.models.image_model import freeze_backbone

    torch.manual_seed(7)
    model = ConvNeXtDetector(pretrained=False, image_size=32)
    freeze_backbone(model)
    stem = model.features[0][0].weight.detach().clone()
    top = model.features[7][0].block[0].weight.detach().clone()
    head = model.classifier[2].weight.detach().clone()
    pixels = torch.randn(2, 3, 32, 32)
    labels = torch.tensor([0, 1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    output = model(pixel_values=pixels, labels=labels)
    assert output.logits.shape == (2, 2)
    assert output.loss.isfinite()
    output.loss.backward()
    optimizer.step()
    assert torch.equal(model.features[0][0].weight, stem)
    assert torch.equal(model.features[7][0].block[0].weight, top)
    assert not torch.equal(model.classifier[2].weight, head)
    unfreeze_top_stages(model, 1)
    optimizer.zero_grad(set_to_none=True)
    model(pixel_values=pixels, labels=labels).loss.backward()
    optimizer.step()
    assert torch.equal(model.features[0][0].weight, stem)
    assert not torch.equal(model.features[7][0].block[0].weight, top)


def test_cnn_checkpoint_reproduces_logits_and_rejects_tampered_weights(tmp_path):
    from veritas.models.cnn_model import ConvNeXtDetector

    torch.manual_seed(9)
    model = ConvNeXtDetector(pretrained=False, image_size=32).eval()
    pixels = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        expected = model(pixel_values=pixels).logits
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "veritas_convnext_tiny"
    assert config["image_size"] == 32
    loaded = ConvNeXtDetector.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(loaded(pixel_values=pixels).logits, expected, rtol=0, atol=0)
    with (tmp_path / "model.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        ConvNeXtDetector.from_pretrained(tmp_path)


def test_cnn_trainer_selects_and_saves_a_real_torch_model(tmp_path):
    from veritas.data.prepare import prepare
    from veritas.training.train_image import TrainImageConfig, train_image

    data = tmp_path / "data"
    prepare(modality="image", output_dir=data, synthetic_subjects=16, synthetic_per_subject=2)
    out = tmp_path / "model"
    metrics = train_image(
        TrainImageConfig(
            data_dir=data,
            output_dir=out,
            architecture="convnext_tiny",
            pretrained=False,
            image_size=32,
            epochs=2,
            batch_size=4,
            freeze_epochs=1,
            unfreeze_stages=1,
            device="cpu",
        )
    )
    doc = json.loads((out / "metrics.json").read_text())
    assert doc["meta"]["architecture"] == "convnext_tiny"
    assert doc["meta"]["data_kind"] == "procedural-fixture"
    from veritas.models.cnn_model import ConvNeXtDetector
    from veritas.models.metadata import torch_state_sha256

    assert doc["meta"]["torch_state_sha256"] == torch_state_sha256(ConvNeXtDetector.from_pretrained(out))
    assert metrics.n_samples > 0
    assert (out / "model.pt").is_file()
    assert (out / "config.json").is_file()
