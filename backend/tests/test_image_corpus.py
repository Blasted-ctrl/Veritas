"""Dataset separation must survive renaming, re-encoding, and generator holdouts."""

from __future__ import annotations

from dataclasses import replace

import pytest

from veritas.data.manifest import Sample, write_manifest

Image = pytest.importorskip("PIL.Image")
PngInfo = pytest.importorskip("PIL.PngImagePlugin").PngInfo


@pytest.fixture
def corpus(tmp_path):
    splits = {}
    for index, split in enumerate(("train", "val", "test")):
        rows = []
        for label in (0, 1):
            path = tmp_path / (split + str(label) + ".png")
            Image.new("RGB", (17, 23), (index * 40 + label * 10, 73, 121)).save(path)
            rows.append(
                Sample(
                    str(path),
                    label,
                    split + str(label),
                    "image",
                    "curated",
                    split,
                    generator_id="training-generator" if label else "",
                )
            )
        splits[split] = rows
    return tmp_path, splits


def save(root, splits):
    for split, rows in splits.items():
        write_manifest(rows, root / (split + ".csv"))


def test_valid_corpus_records_counts_fingerprint_and_provenance(corpus):
    from veritas.data.image_corpus import audit_image_corpus

    root, splits = corpus
    splits["test"][1] = replace(splits["test"][1], generator_id="heldout-generator")
    save(root, splits)
    report = audit_image_corpus(root, heldout_generators=("heldout-generator",))
    assert report["splits"]["train"]["real"] == 1
    assert report["splits"]["test"]["fake"] == 1
    assert len(report["fingerprint"]) == 64
    assert report["heldout_generators"] == ["heldout-generator"]


def test_subject_cannot_cross_dataset_boundaries(corpus):
    from veritas.data.image_corpus import audit_image_corpus

    root, splits = corpus
    splits["val"][0] = replace(splits["val"][0], subject_id=splits["train"][0].subject_id)
    save(root, splits)
    with pytest.raises(ValueError, match="group"):
        audit_image_corpus(root)


def test_reencoded_same_pixels_cannot_cross_dataset_boundaries(corpus):
    from veritas.data.image_corpus import audit_image_corpus

    root, splits = corpus
    first = root / "train0.png"
    second = root / "val0.png"
    metadata = PngInfo()
    metadata.add_text("description", "Different encoding and filename, same photograph")
    with Image.open(first) as image:
        image.save(second, pnginfo=metadata)
    assert first.read_bytes() != second.read_bytes()
    save(root, splits)
    with pytest.raises(ValueError, match="pixels"):
        audit_image_corpus(root)
    with Image.open(second) as image:
        image.putpixel((0, 0), (255, 0, 0))
        image.save(second)
    assert audit_image_corpus(root)["splits"]["val"]["real"] == 1


def test_heldout_generator_cannot_be_present_in_training(corpus):
    from veritas.data.image_corpus import audit_image_corpus

    root, splits = corpus
    for split in ("train", "test"):
        splits[split][1] = replace(splits[split][1], generator_id="heldout-generator")
    save(root, splits)
    with pytest.raises(ValueError, match="held-out"):
        audit_image_corpus(root, heldout_generators=("heldout-generator",))


def test_unknown_holdout_is_not_reported_as_a_success(corpus):
    from veritas.data.image_corpus import audit_image_corpus

    root, splits = corpus
    save(root, splits)
    with pytest.raises(ValueError, match="absent"):
        audit_image_corpus(root, heldout_generators=("not-in-this-dataset",))


def test_single_class_validation_is_rejected(corpus):
    from veritas.data.image_corpus import audit_image_corpus

    root, splits = corpus
    splits["val"] = splits["val"][:1]
    save(root, splits)
    with pytest.raises(ValueError, match="both"):
        audit_image_corpus(root)


def test_split_column_must_agree_with_its_manifest(corpus):
    from veritas.data.image_corpus import audit_image_corpus

    root, splits = corpus
    splits["train"][0] = replace(splits["train"][0], split="test")
    save(root, splits)
    with pytest.raises(ValueError, match="split"):
        audit_image_corpus(root)


def test_procedural_fixture_images_do_not_repeat_across_subjects(tmp_path):
    from veritas.data.synthetic import generate_dataset

    images = generate_dataset(tmp_path, "image", subjects=64, per_subject=2)
    observed = {}
    for sample in images:
        from pathlib import Path

        data = Path(sample.path).read_bytes()
        previous = observed.setdefault(data, sample.subject_id)
        assert previous == sample.subject_id, "fixture pixels repeat across unrelated subject IDs"
