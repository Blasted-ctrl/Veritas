"""Frame decoders release their reader on success, limits, and failures."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
imageio = pytest.importorskip("imageio.v2")
pytest.importorskip("PIL")

from veritas.tasks.video import extract_frame_pngs  # noqa: E402

pytestmark = pytest.mark.api


@contextmanager
def _resource(reader):
    yield SimpleNamespace(legacy_get_reader=lambda: reader)


def test_reader_closes_after_frame_limit(monkeypatch):
    class Reader:
        closed = False

        def get_meta_data(self):
            return {"fps": 4}

        def __iter__(self):
            for value in (20, 40, 60):
                yield np.full((8, 8, 3), value, dtype=np.uint8)

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr(imageio, "imopen", lambda *args, **kwargs: _resource(reader))
    frames = extract_frame_pngs("clip.mp4", fps=4, max_frames=1)
    assert len(frames) == 1
    assert frames[0].startswith(b"\x89PNG\r\n\x1a\n")
    assert reader.closed is True


def test_reader_closes_when_iteration_fails(monkeypatch):
    class Reader:
        closed = False

        def get_meta_data(self):
            return {"fps": 4}

        def __iter__(self):
            yield np.zeros((8, 8, 3), dtype=np.uint8)
            raise RuntimeError("decoder failed mid-stream")

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr(imageio, "imopen", lambda *args, **kwargs: _resource(reader))
    with pytest.raises(RuntimeError, match="mid-stream"):
        extract_frame_pngs("clip.mp4", fps=4)
    assert reader.closed is True


def test_reader_closes_when_frame_conversion_fails(monkeypatch):
    class Reader:
        closed = False

        def get_meta_data(self):
            return {}

        def __iter__(self):
            yield np.zeros((8, 8, 5), dtype=np.uint8)

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr(imageio, "imopen", lambda *args, **kwargs: _resource(reader))
    with pytest.raises(TypeError):
        extract_frame_pngs("clip.gif")
    assert reader.closed is True


def test_missing_fps_metadata_still_extracts_and_closes(monkeypatch):
    class Reader:
        closed = False

        def get_meta_data(self):
            raise ValueError("metadata unavailable")

        def __iter__(self):
            yield np.zeros((8, 8, 3), dtype=np.uint8)
            yield np.full((8, 8, 3), 180, dtype=np.uint8)

        def close(self):
            self.closed = True

    reader = Reader()
    monkeypatch.setattr(imageio, "imopen", lambda *args, **kwargs: _resource(reader))
    assert len(extract_frame_pngs("clip.gif")) == 2
    assert reader.closed is True
