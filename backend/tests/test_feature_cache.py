"""Cached CNN features are bound to actual pixels, prefix weights and split role."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torchvision")
from veritas.data.manifest import Sample, write_manifest  # noqa:E402
from veritas.models.cnn_model import EfficientNetDetector  # noqa:E402
from veritas.training.cached_cnn import ensure_feature_cache  # noqa:E402

pytestmark = pytest.mark.ml


@pytest.fixture
def example(tmp_path):
    torch.set_num_threads(2)
    image = tmp_path / "image.png"
    pixels = np.random.default_rng(8).integers(0, 256, (39, 47, 3), dtype=np.uint8)
    Image.fromarray(pixels).save(image)
    manifest = write_manifest(
        [Sample(str(image), 0, "one", "image", "example", "train")], tmp_path / "train.csv"
    )
    model = EfficientNetDetector(pretrained=False, image_size=32).eval()
    return tmp_path, image, manifest, model


def test_changed_pixels_or_prefix_weights_do_not_reuse_stale_features(example):
    directory, image, manifest, model = example
    first = ensure_feature_cache(model, manifest, directory / "cache", split="train", views=1, seed=3)
    again = ensure_feature_cache(model, manifest, directory / "cache", split="train", views=1, seed=3)
    assert again.path == first.path
    with Image.open(image) as pixels:
        pixels.putpixel((0, 0), (255, 255, 255))
        pixels.save(image)
    changed = ensure_feature_cache(model, manifest, directory / "cache", split="train", views=1, seed=3)
    assert changed.path != first.path
    with torch.no_grad():
        model.features[0][0].weight.add_(0.01)
    weights_changed = ensure_feature_cache(
        model, manifest, directory / "cache", split="train", views=1, seed=3
    )
    assert weights_changed.path != changed.path


def test_corrupted_cache_is_rejected_instead_of_silently_used(example):
    directory, _, manifest, model = example
    cache = ensure_feature_cache(model, manifest, directory / "cache", split="train", views=1, seed=3)
    with cache.path.open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        ensure_feature_cache(model, manifest, directory / "cache", split="train", views=1, seed=3)


def test_sealed_test_rows_cannot_be_used_for_fitting_cache(example):
    directory, image, _, model = example
    manifest = write_manifest(
        [Sample(str(image), 0, "one", "image", "example", "test")], directory / "test.csv"
    )
    with pytest.raises(ValueError, match="split"):
        ensure_feature_cache(model, manifest, directory / "cache", split="train", views=1, seed=3)


def test_validation_processor_changes_invalidate_cache(example, monkeypatch):
    import veritas.models.image_model as processor

    directory, image, _, model = example
    manifest = write_manifest(
        [Sample(str(image), 0, "one", "image", "example", "val")], directory / "val.csv"
    )
    first = ensure_feature_cache(model, manifest, directory / "cache", split="val")
    changed_source = directory / "processor.py"
    changed_source.write_text("# revised validation normalization")
    monkeypatch.setattr(processor, "__file__", str(changed_source))
    changed = ensure_feature_cache(model, manifest, directory / "cache", split="val")
    assert changed.key != first.key


def test_inputs_changing_during_extraction_are_not_published(example, monkeypatch):
    from veritas.training.image_data import ImageManifestDataset

    directory, image, manifest, model = example
    original = ImageManifestDataset.__getitem__

    def changing(self, index):
        result = original(self, index)
        with Image.open(image) as pixels:
            pixels.putpixel((0, 0), (255, 1, 7))
            pixels.save(image)
        return result

    monkeypatch.setattr(ImageManifestDataset, "__getitem__", changing)
    with pytest.raises(ValueError, match="changed during"):
        ensure_feature_cache(model, manifest, directory / "cache", split="train")
    assert not list((directory / "cache").glob("*.json"))
