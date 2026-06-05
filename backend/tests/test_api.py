"""End-to-end tests for the FastAPI inference service.

Builds tiny ViT/Wav2Vec2 models, exports them to ONNX, and drives the /verify
path for image, audio and video uploads. No network and no Redis required (the
cache falls back to memory; Celery runs the video task synchronously).
"""

from __future__ import annotations

import io

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")
pytest.importorskip("imageio")

from fastapi.testclient import TestClient  # noqa: E402

from veritas.api.app import create_app  # noqa: E402
from veritas.api.config import Settings  # noqa: E402
from veritas.data.synthetic import write_png, write_wav  # noqa: E402
from veritas.export.onnx_export import export_audio_onnx, export_image_onnx  # noqa: E402
from veritas.models.audio_model import AudioModelConfig, build_wav2vec2  # noqa: E402
from veritas.models.image_model import ImageModelConfig, build_vit  # noqa: E402

pytestmark = pytest.mark.api

IMAGE_SIZE = 32


@pytest.fixture(scope="module")
def settings(tmp_path_factory) -> Settings:
    d = tmp_path_factory.mktemp("models")
    image_onnx = d / "image.onnx"
    audio_onnx = d / "audio.onnx"
    torch_dir = d / "image_torch"

    vit = build_vit(
        ImageModelConfig(
            pretrained=False,
            image_size=IMAGE_SIZE,
            patch_size=16,
            hidden_size=48,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=96,
        )
    )
    export_image_onnx(vit, image_onnx, image_size=IMAGE_SIZE)
    vit.save_pretrained(torch_dir)  # for Grad-CAM explanation
    export_audio_onnx(
        build_wav2vec2(
            AudioModelConfig(pretrained=False, hidden_size=32, num_hidden_layers=2, num_attention_heads=2)
        ),
        audio_onnx,
        sample_length=4000,
    )
    return Settings(
        image_model=str(image_onnx),
        audio_model=str(audio_onnx),
        image_torch_dir=str(torch_dir),
        image_size=IMAGE_SIZE,
        # Deliberately-unreachable Redis -> exercises the in-memory fallback.
        redis_url="redis://127.0.0.1:6399/0",
        max_video_frames=8,
        # Sample every frame of the tiny test clip so fan-out is exercised.
        video_fps=1000.0,
    )


@pytest.fixture(scope="module")
def client(settings) -> TestClient:
    return TestClient(create_app(settings))


def _png_bytes(tmp_path, name="x.png") -> bytes:
    p = tmp_path / name
    write_png(p, [[(i * 7 % 256, 0, 0) for i in range(IMAGE_SIZE)] for _ in range(IMAGE_SIZE)])
    return p.read_bytes()


def _wav_bytes(tmp_path, name="x.wav") -> bytes:
    p = tmp_path / name
    write_wav(p, [int(1000 * ((i % 32) - 16)) for i in range(4000)])
    return p.read_bytes()


def _gif_bytes() -> bytes:
    from PIL import Image

    frames = [Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (i * 40 % 256, 10, 10)) for i in range(4)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    return buf.getvalue()


def test_health(client, settings):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["image_model"] is True
    assert body["audio_model"] is True
    assert body["cache"] == "memory"  # unreachable redis -> fallback


def test_verify_image(client, tmp_path):
    r = client.post("/verify", files={"file": ("x.png", _png_bytes(tmp_path), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modality"] == "image"
    assert body["verdict"] in {"real", "fake"}
    assert 0.0 <= body["fake_probability"] <= 1.0
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["latency_ms"] >= 0.0
    assert len(body["content_sha256"]) == 64


def test_verify_image_with_explain_returns_heatmap(client, tmp_path):
    r = client.post(
        "/verify?explain=true",
        files={"file": ("e.png", _png_bytes(tmp_path, "explain.png"), "image/png")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modality"] == "image"
    assert body["heatmap"].startswith("data:image/png;base64,")
    assert body["overlay"].startswith("data:image/png;base64,")


def test_health_reports_explainer(client):
    assert client.get("/health").json()["explainer"] is True


def test_verify_audio(client, tmp_path):
    r = client.post("/verify", files={"file": ("x.wav", _wav_bytes(tmp_path), "audio/wav")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modality"] == "audio"
    assert body["verdict"] in {"real", "fake"}


def test_verify_video_fans_out_to_frames(client):
    r = client.post("/verify", files={"file": ("clip.gif", _gif_bytes(), "image/gif")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["modality"] == "video"
    assert body["frames_analyzed"] == 4
    assert len(body["frames"]) == 4
    assert all(0.0 <= f["fake_probability"] <= 1.0 for f in body["frames"])


def test_cache_hit_on_repeated_upload(settings, tmp_path):
    # Fresh app => isolated (empty) cache, independent of other tests.
    c = TestClient(create_app(settings))
    payload = _png_bytes(tmp_path, "cacheme.png")
    first = c.post("/verify", files={"file": ("c.png", payload, "image/png")}).json()
    second = c.post("/verify", files={"file": ("c.png", payload, "image/png")}).json()
    assert first["cached"] is False
    assert second["cached"] is True
    assert first["content_sha256"] == second["content_sha256"]
    assert first["fake_probability"] == second["fake_probability"]


def test_unsupported_media_type(client):
    r = client.post("/verify", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_empty_upload_rejected(client):
    r = client.post("/verify", files={"file": ("x.png", b"", "image/png")})
    assert r.status_code == 400


def test_missing_model_returns_503(tmp_path):
    bad = Settings(image_model=str(tmp_path / "nope.onnx"), audio_model=str(tmp_path / "nope2.onnx"))
    c = TestClient(create_app(bad))
    r = c.post("/verify", files={"file": ("x.png", _png_bytes(tmp_path), "image/png")})
    assert r.status_code == 503
