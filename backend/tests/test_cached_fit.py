"""Actual CPU fitting writes validation-only completion; test data stays out."""

import json

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torchvision")
pytest.importorskip("sklearn")
from veritas.data.manifest import Sample, write_manifest  # noqa:E402
from veritas.models.cnn_model import EfficientNetDetector  # noqa:E402
from veritas.training.cached_cnn import CachedTrainConfig, fit_cached  # noqa:E402

pytestmark = pytest.mark.ml


def test_real_cached_fit_saves_bound_validation_checkpoint_without_opening_test(tmp_path):
    torch.set_num_threads(2)
    for split in ("train", "val"):
        rows = []
        for label in (0, 1):
            image = tmp_path / f"{split}-{label}.png"
            pixels = np.random.default_rng(label + (10 if split == "val" else 0)).integers(
                0, 256, (37, 43, 3), dtype=np.uint8
            )
            Image.fromarray(pixels).save(image)
            rows.append(Sample(str(image), label, f"{split}-{label}", "image", "curated", split))
        write_manifest(rows, tmp_path / (split + ".csv"))
    (tmp_path / "test.csv").write_text("This is sealed and not a valid manifest. Fitting must not read it.")
    model = EfficientNetDetector(pretrained=False, image_size=32)
    result = fit_cached(
        model,
        tmp_path / "train.csv",
        tmp_path / "val.csv",
        tmp_path / "model",
        tmp_path / "cache",
        CachedTrainConfig(epochs=2, batch_size=2),
        data_audit={"fingerprint": "a" * 64, "data_kind": "curated-images"},
    )
    saved = EfficientNetDetector.from_pretrained(tmp_path / "model")
    metrics = json.loads((tmp_path / "model" / "metrics.json").read_text())
    assert metrics["meta"]["evaluation_split"] == "val"
    assert metrics["meta"]["test_evaluated"] is False
    assert metrics["meta"]["dataset_fingerprint"] == "a" * 64
    assert saved.config.model_type == "veritas_efficientnet_b0"
    assert 0 <= result["best_partial_auc"] <= 1
    assert len(result["history"]) == 2
