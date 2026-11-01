"""API contracts exercised with real, input-dependent ONNX graphs, without Torch."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
pytest.importorskip("fastapi")
pytest.importorskip("pydantic_settings")
Image = pytest.importorskip("PIL.Image")

from fastapi import HTTPException, UploadFile  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from veritas.api.app import create_app  # noqa: E402
from veritas.api.config import Settings  # noqa: E402
from veritas.models.metadata import file_sha256, metadata_path  # noqa: E402

pytestmark = pytest.mark.api
IMAGE_SIZE = 32


def _write_model(path: Path, *, sign: float = 1.0, classes: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [None, 3, 32, 32])]
    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, classes])]
    nodes = [
        helper.make_node("ReduceMean", ["pixel_values"], ["mean"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Flatten", ["mean"], ["flat"], axis=1),
        helper.make_node("Mul", ["flat", "sign"], ["positive"]),
        helper.make_node("Neg", ["positive"], ["negative"]),
        helper.make_node(
            "Concat", ["negative", "positive"] + ["positive"] * (classes - 2), ["logits"], axis=1
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "contract-fixture",
        inputs,
        outputs,
        [numpy_helper.from_array(np.array(sign, dtype=np.float32), name="sign")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    _write_metadata(path)
    return path


def _write_metadata(path: Path, **overrides) -> None:
    info = {
        "format_version": 1,
        "id": "contract-fixture/detector",
        "source_type": "custom",
        "artifact_sha256": file_sha256(path),
        "image_size": IMAGE_SIZE,
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "resample": 2,
        "rescale_factor": 1 / 255,
        "fake_index": 1,
    }
    info.update(overrides)
    metadata_path(path).write_text(json.dumps(info), encoding="utf-8")


@pytest.fixture
def settings(tmp_path):
    return Settings(
        image_model=str(_write_model(tmp_path / "model.onnx")),
        audio_model=str(tmp_path / "missing-audio.onnx"),
        image_torch_dir=str(tmp_path / "missing-explainer"),
        image_size=IMAGE_SIZE,
        allowed_origins=["http://localhost:3000"],
        fake_threshold=0.5,
        max_upload_bytes=26_214_400,
        cache_ttl_seconds=3600,
        redis_url="",
        video_fps=1000.0,
        max_video_frames=8,
    )


@pytest.fixture
def shared_redis(monkeypatch):
    values: dict[str, bytes] = {}

    class SharedRedis:
        def ping(self):
            return True

        def get(self, key):
            return values.get(key)

        def setex(self, key, _ttl, value):
            values[key] = value.encode("utf-8")

    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(Redis=SimpleNamespace(from_url=lambda *args, **kwargs: SharedRedis())),
    )
    return "redis://shared-cache.test:6379/0"


def _client(settings):
    return TestClient(create_app(settings), raise_server_exceptions=False)


def _png_bytes(value: int = 240) -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (39, 17), (value, value, value)).save(stream, format="PNG")
    return stream.getvalue()


def _gif_bytes() -> bytes:
    stream = io.BytesIO()
    frames = [Image.new("RGB", (32, 32), (v, v, v)) for v in (10, 90, 170, 250)]
    frames[0].save(stream, format="GIF", save_all=True, append_images=frames[1:], duration=250, loop=0)
    return stream.getvalue()


def _post(client, payload: bytes, *, name="image.png", content_type="image/png"):
    return client.post("/verify", files={"file": (name, payload, content_type)})


def test_allowed_browser_origin_receives_cors_headers(settings):
    client = _client(settings)
    origin = "http://localhost:3000"
    response = client.get("/health", headers={"Origin": origin})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == origin
    preflight = client.options(
        "/verify",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers.get("access-control-allow-origin") == origin


def test_unconfigured_origin_does_not_receive_permission(settings):
    response = _client(settings).get("/health", headers={"Origin": "https://unconfigured.example"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("artifact", ["corrupt", "three-classes"])
def test_unusable_graph_is_unavailable_in_health_and_verification(settings, tmp_path, artifact):
    path = tmp_path / "unusable.onnx"
    if artifact == "corrupt":
        path.write_bytes(b"not an ONNX graph")
    else:
        _write_model(path, classes=3)
    client = _client(settings.model_copy(update={"image_model": str(path)}))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["image_model"] is False
    assert _post(client, _png_bytes()).status_code == 503


def test_pretrained_provenance_survives_cached_results(settings):
    model = Path(settings.image_model)
    _write_metadata(
        model,
        source_type="pretrained",
        revision="1" * 40,
        source_url="https://example.com/models/contract-fixture",
        license="Apache-2.0",
    )
    client = _client(settings)
    first = _post(client, _png_bytes())
    second = _post(client, _png_bytes())
    assert first.status_code == second.status_code == 200
    info = first.json()["model_info"]
    assert info["id"] == "contract-fixture/detector"
    assert info["source_type"] == "pretrained"
    assert info["revision"] == "1" * 40
    assert info["source_url"] == "https://example.com/models/contract-fixture"
    assert info["artifact_sha256"] == file_sha256(model)
    assert second.json()["model_info"] == info
    assert first.json()["cached"] is False
    assert second.json()["cached"] is True


def test_custom_model_is_not_attributed_to_default_pretrained_checkpoint(settings):
    metadata_path(settings.image_model).unlink()
    response = _post(_client(settings), _png_bytes())
    assert response.status_code == 200
    assert response.json()["model_info"]["source_type"] == "custom"
    assert not response.json()["model_info"].get("source_url")


def test_reversed_class_mapping_controls_real_inference(settings):
    model = Path(settings.image_model)
    _write_metadata(model, fake_index=0)
    response = _post(_client(settings), _png_bytes())
    assert response.status_code == 200
    assert response.json()["fake_probability"] < 0.2
    assert response.json()["verdict"] == "real"


def test_size_limit_rejects_one_byte_over_but_accepts_boundary(settings):
    payload = _png_bytes()
    exact = _client(settings.model_copy(update={"max_upload_bytes": len(payload)}))
    over = _client(settings.model_copy(update={"max_upload_bytes": len(payload) - 1}))
    assert _post(exact, payload).status_code == 200
    assert _post(over, payload).status_code == 413


def test_oversize_upload_is_rejected_without_reading_the_whole_stream(settings):
    limit = 64

    class BoundedStream(io.BytesIO):
        consumed = 0

        def read(self, size=-1):
            assert 0 <= size <= limit + 1, "upload reader requested an unbounded allocation"
            value = super().read(size)
            self.consumed += len(value)
            assert self.consumed <= limit + 1, "upload reader consumed the whole oversized file"
            return value

    app = create_app(settings.model_copy(update={"max_upload_bytes": limit}))
    endpoint = next(route.endpoint for route in app.routes if getattr(route, "path", None) == "/verify")
    stream = BoundedStream(b"x" * 8192)
    try:
        with pytest.raises(HTTPException) as rejected:
            asyncio.run(endpoint(file=UploadFile(file=stream, filename="too-large.png"), explain=False))
        assert rejected.value.status_code == 413
        assert stream.consumed <= limit + 1
    finally:
        stream.close()


def test_corrupt_video_returns_422(settings):
    response = _post(_client(settings), b"not a video", name="broken.mp4", content_type="video/mp4")
    assert response.status_code == 422
    assert "verdict" not in response.json()


def test_zero_frame_analysis_is_not_cached_and_temporary_upload_is_removed(settings, monkeypatch):
    import veritas.tasks.video as video

    client = _client(settings)
    payload = _gif_bytes()
    captured: list[Path] = []
    original = video.extract_frame_pngs

    def no_frames(path, **kwargs):
        captured.append(Path(path))
        assert captured[-1].exists()
        return []

    monkeypatch.setattr(video, "extract_frame_pngs", no_frames)
    failed = _post(client, payload, name="clip.gif", content_type="image/gif")
    assert failed.status_code == 422
    assert captured and not captured[0].exists()
    monkeypatch.setattr(video, "extract_frame_pngs", original)
    recovered = _post(client, payload, name="clip.gif", content_type="image/gif")
    assert recovered.status_code == 200
    assert recovered.json()["frames_analyzed"] == 4
    assert recovered.json()["cached"] is False


def test_decoder_failure_removes_temporary_upload(settings, monkeypatch):
    import veritas.tasks.video as video

    captured: list[Path] = []

    def broken_decoder(path, **kwargs):
        captured.append(Path(path))
        assert captured[-1].exists()
        raise ValueError("cannot decode frames")

    monkeypatch.setattr(video, "extract_frame_pngs", broken_decoder)
    response = _post(_client(settings), _gif_bytes(), name="clip.gif", content_type="image/gif")
    assert response.status_code == 422
    assert captured and not captured[0].exists()


def test_cache_isolated_between_model_artifacts(settings, tmp_path, shared_redis):
    first_client = _client(settings.model_copy(update={"redis_url": shared_redis}))
    second_model = _write_model(tmp_path / "other.onnx", sign=-1.0)
    second_client = _client(
        settings.model_copy(
            update={
                "redis_url": shared_redis,
                "image_model": str(second_model),
            }
        )
    )
    first = _post(first_client, _png_bytes()).json()
    second = _post(second_client, _png_bytes()).json()
    assert first["fake_probability"] > 0.8
    assert second["fake_probability"] < 0.2
    assert second["cached"] is False


def test_cache_isolated_between_processor_contracts(settings, shared_redis):
    first_client = _client(settings.model_copy(update={"redis_url": shared_redis}))
    first = _post(first_client, _png_bytes()).json()
    _write_metadata(Path(settings.image_model), mean=[1.0, 1.0, 1.0])
    second_client = _client(settings.model_copy(update={"redis_url": shared_redis}))
    second = _post(second_client, _png_bytes()).json()
    assert first["verdict"] == "fake"
    assert second["verdict"] == "real"
    assert second["cached"] is False


def test_cache_isolated_between_verdict_thresholds(settings, shared_redis):
    first_client = _client(settings.model_copy(update={"redis_url": shared_redis, "fake_threshold": 0.5}))
    second_client = _client(settings.model_copy(update={"redis_url": shared_redis, "fake_threshold": 0.99}))
    first = _post(first_client, _png_bytes()).json()
    second = _post(second_client, _png_bytes()).json()
    assert first["verdict"] == "fake"
    assert second["verdict"] == "real"
    assert second["cached"] is False


def test_cache_isolated_between_media_interpretations(settings):
    client = _client(settings)
    payload = _gif_bytes()
    image = _post(client, payload, name="first-frame.png", content_type="image/png")
    video = _post(client, payload, name="clip.gif", content_type="image/gif")
    assert image.status_code == video.status_code == 200
    assert image.json()["modality"] == "image"
    assert video.json()["modality"] == "video"
    assert video.json()["frames_analyzed"] == 4
    assert video.json()["cached"] is False


def test_cache_isolated_between_video_frame_caps(settings, shared_redis):
    first_client = _client(settings.model_copy(update={"redis_url": shared_redis, "max_video_frames": 1}))
    second_client = _client(settings.model_copy(update={"redis_url": shared_redis, "max_video_frames": 3}))
    payload = _gif_bytes()
    first = _post(first_client, payload, name="clip.gif", content_type="image/gif").json()
    second = _post(second_client, payload, name="clip.gif", content_type="image/gif").json()
    assert first["frames_analyzed"] == 1
    assert second["frames_analyzed"] == 3
    assert second["cached"] is False


def test_real_mp4_sampling_and_cache_fps_isolation(settings, tmp_path, shared_redis):
    pytest.importorskip("imageio_ffmpeg")
    imageio = pytest.importorskip("imageio.v2")
    clip = tmp_path / "four-frames.mp4"
    with imageio.get_writer(clip, fps=4, codec="libx264", macro_block_size=None) as writer:
        for value in (10, 90, 170, 250):
            writer.append_data(np.full((32, 32, 3), value, dtype=np.uint8))
    payload = clip.read_bytes()
    first_client = _client(settings.model_copy(update={"redis_url": shared_redis, "video_fps": 1.0}))
    second_client = _client(settings.model_copy(update={"redis_url": shared_redis, "video_fps": 2.0}))
    first_response = _post(first_client, payload, name="clip.mp4", content_type="video/mp4")
    second_response = _post(second_client, payload, name="clip.mp4", content_type="video/mp4")
    assert first_response.status_code == second_response.status_code == 200
    first, second = first_response.json(), second_response.json()
    assert first["frames_analyzed"] == 1
    assert second["frames_analyzed"] == 2
    assert second["cached"] is False
    mean = sum(frame["fake_probability"] for frame in second["frames"]) / 2
    assert second["fake_probability"] == pytest.approx(mean)
