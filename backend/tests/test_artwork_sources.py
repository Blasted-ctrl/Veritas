"""Origin labels and related-image groups must survive artwork preparation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def artwork(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "prepare_artwork_pilot", scripts / "prepare_artwork_pilot.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_rows(artwork, monkeypatch, tmp_path, kind, rows):
    (tmp_path / "metadata").mkdir(exist_ok=True)
    revision = artwork.SOURCES[kind][1]
    monkeypatch.setattr(artwork, "download", lambda *args, **kwargs: json.dumps({"sha": revision}).encode())
    document = {
        "rows": [
            {
                "row_idx": index,
                "row": row
                | {
                    "image": {
                        "src": f"https://datasets-server.huggingface.co/cached-assets/source/--/{revision}/--/default/train/{index}/image.jpg"
                    }
                },
            }
            for index, row in enumerate(rows)
        ]
    }
    monkeypatch.setattr(artwork, "cached_json", lambda *args: document)
    return artwork.hf_rows(tmp_path, kind, [0])


def test_human_art_is_not_relabeled_by_generated_caption(artwork, monkeypatch, tmp_path):
    rows = source_rows(
        artwork,
        monkeypatch,
        tmp_path,
        "artbench",
        [
            {
                "artist": "artist-a",
                "url": "https://example.org/old-drawing",
                "caption": "AI-looking illustrated image",
            },
            {"artist": "artist-a", "url": "https://example.org/old-painting", "caption": "A painting"},
        ],
    )
    assert [row["label"] for row in rows] == [0, 0]
    assert rows[0]["group"] == rows[1]["group"] == "artist:artist-a"
    assert artwork.group_role(rows[0]["group"]) == artwork.group_role(rows[1]["group"])


def test_ai_hand_drawn_style_stays_ai_and_prompt_variants_stay_together(artwork, monkeypatch, tmp_path):
    rows = source_rows(
        artwork,
        monkeypatch,
        tmp_path,
        "lunara",
        [
            {"prompt": "Hand-drawn anime hero", "model": "Generator A"},
            {"prompt": "  HAND-DRAWN   ANIME hero ", "model": "Generator B"},
        ],
    )
    assert [row["label"] for row in rows] == [1, 1]
    assert {row["domain"] for row in rows} == {"illustration"}
    assert rows[0]["group"] == rows[1]["group"]
    assert artwork.group_role(rows[0]["group"]) == artwork.group_role(rows[1]["group"])


def test_changed_source_revision_is_rejected(artwork, monkeypatch, tmp_path):
    monkeypatch.setattr(artwork, "download", lambda *args, **kwargs: b'{"sha":"different"}')
    with pytest.raises(ValueError, match="head changed"):
        artwork.hf_rows(tmp_path, "artbench", [0])


@pytest.mark.parametrize("changed", [False, True])
def test_expired_viewer_links_refresh_without_changing_origin(artwork, monkeypatch, tmp_path, changed):
    path = tmp_path / "page.json"
    old = {
        "rows": [
            {
                "row_idx": 4,
                "row": {
                    "prompt": "anime hero",
                    "model": "A",
                    "image": {"src": "https://example.org/pinned/4.jpg?Expires=1"},
                },
            }
        ]
    }
    fresh = json.loads(json.dumps(old))
    fresh["rows"][0]["row"]["image"]["src"] = "https://example.org/pinned/4.jpg?Expires=9999999999"
    if changed:
        fresh["rows"][0]["row"]["model"] = "B"
    path.write_text(json.dumps(old))
    monkeypatch.setattr(artwork, "pace_metadata", lambda: None)
    monkeypatch.setattr(artwork, "download", lambda *args, **kwargs: json.dumps(fresh).encode())
    if changed:
        with pytest.raises(ValueError, match="source identity changed"):
            artwork.hf_document("https://example.org/page", path)
        assert json.loads(path.read_text()) == old
    else:
        assert artwork.hf_document("https://example.org/page", path) == fresh
        assert json.loads(path.read_text()) == fresh


def test_resumed_anime_acquisition_checks_live_version(artwork, monkeypatch, tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / "metadata" / "pixiv-info.json").write_text('{"currentVersionNumber":1}')
    monkeypatch.setattr(artwork, "download", lambda *args, **kwargs: b'{"currentVersionNumber":2}')
    with pytest.raises(ValueError, match="Pixiv source version changed"):
        artwork.anime_rows(tmp_path)


@pytest.mark.ml
def test_training_rejects_a_parent_checkpoint_outside_the_protocol(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_diverse_cnn.py"
    spec = importlib.util.spec_from_file_location("train_diverse_cnn", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "model.pt").write_bytes(b"unregistered checkpoint")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"training": {"initial_checkpoint_sha256": "0" * 64}}))
    monkeypatch.setattr(
        "sys.argv",
        [
            str(script),
            "--data-dir",
            str(tmp_path / "absent-data"),
            "--output-dir",
            str(tmp_path / "output"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--protocol",
            str(protocol),
            "--initialize-from",
            str(parent),
        ],
    )
    with pytest.raises(ValueError, match="Initial checkpoint differs"):
        module.main()
