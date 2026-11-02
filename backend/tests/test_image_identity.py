"""Content identities stay correct while avoiding repeated image decoding."""

from __future__ import annotations

import hashlib
import io
import struct

import pytest

Image = pytest.importorskip("PIL.Image")
from veritas.data.image_corpus import image_identity  # noqa: E402

pytestmark = pytest.mark.api


def image_bytes(tmp_path, name):
    pixels = (hashlib.sha256(str(tmp_path / name).encode()).digest() * 2)[:48]
    image = Image.frombytes("RGB", (4, 4), pixels)
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_unchanged_copies_reuse_decode_but_changed_bytes_do_not(tmp_path, monkeypatch):
    first, copy = tmp_path / "first.png", tmp_path / "copy.png"
    first.write_bytes(image_bytes(tmp_path, "first"))
    copy.write_bytes(first.read_bytes())
    original = Image.open
    calls = []

    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(Image, "open", tracked)
    before = image_identity(first)
    assert image_identity(first) == image_identity(copy) == before
    assert len(calls) == 1
    first.write_bytes(image_bytes(tmp_path, "changed"))
    after = image_identity(first)
    assert after[0] != before[0] and after[1] != before[1]
    assert len(calls) == 2


def test_encoded_and_decoded_hashes_describe_the_same_byte_snapshot(tmp_path, monkeypatch):
    path = tmp_path / "changing.png"
    original_bytes = image_bytes(tmp_path, "original")
    replacement = image_bytes(tmp_path, "replacement")
    path.write_bytes(original_bytes)
    with Image.open(io.BytesIO(original_bytes)) as image:
        rgb = image.convert("RGB")
        expected_pixels = hashlib.sha256(struct.pack("!II", *rgb.size) + rgb.tobytes()).hexdigest()
    original_convert = Image.Image.convert

    def changing(self, *args, **kwargs):
        result = original_convert(self, *args, **kwargs)
        path.write_bytes(replacement)
        return result

    monkeypatch.setattr(Image.Image, "convert", changing)
    assert image_identity(path) == (hashlib.sha256(original_bytes).hexdigest(), expected_pixels)
    assert path.read_bytes() == replacement
