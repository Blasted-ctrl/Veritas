"""Tests for dataset adapters using small on-disk fixtures."""

from __future__ import annotations

import json

import pytest

from veritas.data import datasets
from veritas.data.manifest import FAKE, REAL
from veritas.data.synthetic import write_png, write_wav


def _touch_png(path):
    write_png(path, [[(1, 2, 3)]])


# --- generic directory --------------------------------------------------------
def test_from_directory_groups_subject_from_filename(tmp_path):
    _touch_png(tmp_path / "real" / "subject_0001_00.png")
    _touch_png(tmp_path / "real" / "subject_0001_01.png")
    _touch_png(tmp_path / "fake" / "subject_0001_02.png")
    _touch_png(tmp_path / "fake" / "subject_0002_00.png")

    samples = datasets.from_directory(tmp_path, "image")
    assert len(samples) == 4
    by_subject = {s.subject_id for s in samples}
    assert by_subject == {"subject_0001", "subject_0002"}
    # subject_0001 has both real and fake samples (the leakage-prone case).
    s1_labels = {s.label for s in samples if s.subject_id == "subject_0001"}
    assert s1_labels == {REAL, FAKE}


def test_from_directory_errors_when_empty(tmp_path):
    (tmp_path / "real").mkdir()
    with pytest.raises(FileNotFoundError):
        datasets.from_directory(tmp_path, "image")


# --- FaceForensics++ ----------------------------------------------------------
def test_from_faceforensics_layout(tmp_path):
    real = tmp_path / "original_sequences" / "youtube" / "c23" / "frames" / "033"
    _touch_png(real / "0000.png")
    _touch_png(real / "0001.png")
    fake = tmp_path / "manipulated_sequences" / "Deepfakes" / "c23" / "frames" / "033_097"
    _touch_png(fake / "0000.png")

    samples = datasets.from_faceforensics(tmp_path)
    assert len(samples) == 3
    # The manipulated clip's target identity (033) groups with the real 033.
    assert all(s.subject_id == "ffpp_033" for s in samples)
    assert {s.label for s in samples} == {REAL, FAKE}


# --- DFDC ---------------------------------------------------------------------
def test_from_dfdc_metadata(tmp_path):
    part = tmp_path / "dfdc_part_0"
    meta = {
        "abc.mp4": {"label": "FAKE", "original": "src.mp4"},
        "src.mp4": {"label": "REAL", "original": None},
    }
    (part).mkdir(parents=True)
    (part / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    _touch_png(part / "abc" / "0.png")
    _touch_png(part / "src" / "0.png")

    samples = datasets.from_dfdc(tmp_path)
    assert len(samples) == 2
    # The fake derived from src.mp4 shares src's subject id.
    assert {s.subject_id for s in samples} == {"dfdc_src"}
    assert {s.label for s in samples} == {REAL, FAKE}


# --- ASVspoof -----------------------------------------------------------------
def test_from_asvspoof_protocol(tmp_path):
    write_wav(tmp_path / "flac" / "LA_T_1000001.wav", [0, 1, 2])
    write_wav(tmp_path / "flac" / "LA_T_1000002.wav", [0, 1, 2])
    protocol = tmp_path / "ASVspoof2019.LA.cm.train.trn.txt"
    protocol.write_text(
        "LA_0079 LA_T_1000001 - - bonafide\nLA_0079 LA_T_1000002 - A01 spoof\n",
        encoding="utf-8",
    )
    samples = datasets.from_asvspoof(tmp_path, protocol=protocol)
    assert len(samples) == 2
    assert {s.subject_id for s in samples} == {"asv_LA_0079"}
    assert {s.label for s in samples} == {REAL, FAKE}


def test_load_source_unknown():
    with pytest.raises(ValueError):
        datasets.load_source("nope", "image", "/tmp")
