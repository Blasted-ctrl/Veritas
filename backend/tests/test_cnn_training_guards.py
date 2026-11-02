"""Training limits and exported processors must preserve completed provenance."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("torchvision")
pytest.importorskip("transformers")
pytest.importorskip("sklearn")

from veritas.data.manifest import Sample, read_manifest, write_manifest  # noqa: E402
from veritas.models.metadata import file_sha256, metadata_path, torch_state_sha256  # noqa: E402
from veritas.training.image_data import ImageManifestDataset  # noqa: E402
from veritas.training.train_image import TrainImageConfig, train_image  # noqa: E402


@pytest.fixture(autouse=True)
def two_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def corpus_with_unselected_holdout(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    splits = {}
    for split_index, split in enumerate(("train", "val", "test")):
        rows = []
        for index, label in enumerate((0, 1, 1)):
            name = f"{split}-{index}"
            path = data_dir / f"{name}.png"
            # Every file has distinct decoded pixels, including across splits.
            Image.new("RGB", (35, 37), (split_index * 60 + index * 15, 73, 121)).save(path)
            rows.append(
                Sample(
                    path=str(path),
                    label=label,
                    subject_id=name,
                    modality="image",
                    source="curated",
                    split=split,
                    generator_id="training-generator" if label else "",
                )
            )
        splits[split] = rows
        write_manifest(rows, data_dir / f"{split}.csv")

    selected = ImageManifestDataset(data_dir / "test.csv", image_size=32, limit=2, seed=1337)
    selected_paths = {sample.path for sample in selected.samples}
    excluded = next(
        sample for sample in splits["test"] if sample.label == 1 and sample.path not in selected_paths
    )
    heldout = "test-only-generator"
    splits["test"] = [
        replace(sample, generator_id=heldout) if sample == excluded else sample for sample in splits["test"]
    ]
    write_manifest(splits["test"], data_dir / "test.csv")
    return data_dir, heldout


def _training_config(data_dir, output_dir, heldout, *, limit=None):
    return TrainImageConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        architecture="vit",
        pretrained=False,
        image_size=32,
        epochs=1,
        batch_size=3,
        freeze_epochs=1,
        seed=1337,
        device="cpu",
        heldout_generators=(heldout,),
        limit=limit,
    )


@pytest.mark.ml
def test_limit_cannot_silently_remove_requested_heldout_family(corpus_with_unselected_holdout, tmp_path):
    data_dir, heldout = corpus_with_unselected_holdout
    selected = ImageManifestDataset(data_dir / "test.csv", image_size=32, limit=2, seed=1337)
    assert all(sample.generator_id != heldout for sample in selected.samples)
    output_dir = tmp_path / "limited-model"

    with pytest.raises(ValueError, match=r"(?i)(held.?out|selected|limit)"):
        train_image(_training_config(data_dir, output_dir, heldout, limit=2))

    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "metrics.json").exists()


@pytest.mark.ml
@pytest.mark.parametrize("padded", [False, True])
def test_full_test_split_evaluates_requested_heldout_family(corpus_with_unselected_holdout, tmp_path, padded):
    data_dir, heldout = corpus_with_unselected_holdout
    if padded:
        manifest = data_dir / "test.csv"
        write_manifest(
            [replace(row, generator_id=" " + row.generator_id + " ") for row in read_manifest(manifest)],
            manifest,
        )
    output_dir = tmp_path / "complete-model"

    train_image(_training_config(data_dir, output_dir, heldout))

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((output_dir / "data-audit.json").read_text(encoding="utf-8"))
    assert metrics["meta"]["heldout_generators"] == [heldout]
    assert metrics["meta"]["test_size"] == metrics["n_samples"] == 3
    assert metrics["n_positive"] == 2
    assert metrics["n_negative"] == 1
    assert audit["splits"]["test"]["generators"][heldout] == 1
    assert metrics["meta"]["dataset_fingerprint"] == audit["fingerprint"]
    assert (output_dir / "config.json").exists()


@pytest.mark.api
def test_cnn_export_rejects_processor_changed_after_completed_training(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from veritas.export.onnx_export import export_image_from_dir
    from veritas.models.cnn_model import ConvNeXtDetector

    model_dir = tmp_path / "checkpoint"
    model = ConvNeXtDetector(pretrained=False, image_size=32).eval()
    model.save_pretrained(model_dir)
    fingerprint = "a" * 64
    (model_dir / "data-audit.json").write_text(
        json.dumps({"fingerprint": fingerprint, "data_kind": "curated-images"}), encoding="utf-8"
    )
    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "meta": {
                    "dataset_fingerprint": fingerprint,
                    "torch_state_sha256": torch_state_sha256(model),
                    "image_size": 32,
                }
            }
        ),
        encoding="utf-8",
    )
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    weights_hash = file_sha256(model_dir / "model.pt")
    config["image_size"] = 64
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert config["checkpoint_sha256"] == file_sha256(model_dir / "model.pt") == weights_hash
    output_path = tmp_path / "image.onnx"

    with pytest.raises(ValueError):
        export_image_from_dir(model_dir, output_path)

    assert not output_path.exists()
    assert not metadata_path(output_path).exists()
