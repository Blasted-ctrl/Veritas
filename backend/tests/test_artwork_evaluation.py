"""Artwork evaluation rejects changed pixels, role leaks and diagnostic reuse."""

import importlib.util
import json
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")
pytest.importorskip("numpy")
pytest.importorskip("sklearn")
from veritas.data.image_corpus import image_identity  # noqa: E402
from veritas.data.manifest import Sample, write_manifest  # noqa: E402
from veritas.models.metadata import file_sha256  # noqa: E402

pytestmark = pytest.mark.ml


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("evaluate_artwork_cnn", scripts / "evaluate_artwork_cnn.py")
    assert spec is not None and spec.loader is not None
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    source, records = {"manifest_sha256": {}}, []
    for number, role in enumerate(("train", "val", "calibration", "test")):
        image = tmp_path / (role + ".png")
        Image.new("RGB", (17, 19), (number * 30, 10, 20)).save(image)
        group = role + "-group"
        manifest = write_manifest(
            [Sample(str(image), 0, group, "image", "artwork", role if role in ("train", "val") else "test")],
            tmp_path / (role + ".csv"),
        )
        raw, pixels = image_identity(image)
        records.append(
            {
                "prepared_sha256": raw,
                "downloaded_sha256": raw,
                "decoded_sha256": pixels,
                "group": group,
                "role": role,
                "label": 0,
            }
        )
        source["manifest_sha256"][role] = file_sha256(manifest)
    (tmp_path / "provenance.json").write_text(json.dumps(records))
    return evaluation, source, records


def test_artwork_pixels_are_bound_before_scoring(corpus, tmp_path):
    evaluation, source, _ = corpus
    evaluation.verify_roles(tmp_path, source, "f" * 64)
    Image.new("RGB", (17, 19), (255, 100, 10)).save(tmp_path / "test.png")
    with pytest.raises((ValueError, KeyError)):
        evaluation.verify_roles(tmp_path, source, "f" * 64)


def test_broly_diagnostic_cannot_enter_training_or_test(corpus, tmp_path):
    evaluation, source, records = corpus
    for record in records:
        with pytest.raises(ValueError, match="Diagnostic image leaked"):
            evaluation.verify_roles(tmp_path, source, record["downloaded_sha256"])


def test_artwork_groups_cannot_cross_calibration_and_test(corpus, tmp_path):
    evaluation, source, records = corpus
    records[-1]["group"] = records[-2]["group"]
    write_manifest(
        [Sample(str(tmp_path / "test.png"), 0, records[-1]["group"], "image", "artwork", "test")],
        tmp_path / "test.csv",
    )
    source["manifest_sha256"]["test"] = file_sha256(tmp_path / "test.csv")
    (tmp_path / "provenance.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="Group crosses"):
        evaluation.verify_roles(tmp_path, source, "f" * 64)
