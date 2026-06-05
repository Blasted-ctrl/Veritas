"""FastAPI inference service: POST /verify returns an authenticity verdict.

Routing by modality:
  * image -> ONNX ViT detector
  * audio -> ONNX Wav2Vec2 detector
  * video -> Celery task that fans out to frames and aggregates

Results are cached by content hash; inference latency is measured and returned
(and logged). The app degrades gracefully: missing models yield 503, an
unreachable Redis falls back to an in-memory cache.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from veritas.api.cache import ResultCache, content_hash
from veritas.api.config import Settings, get_settings
from veritas.api.inference import AudioDetector, ImageDetector, verdict_from_probability
from veritas.api.schemas import HealthResponse, Verdict

logger = logging.getLogger("veritas.api")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".gif"}


def _modality_for(filename: str, content_type: str | None) -> str | None:
    # Extension wins over content-type: an (animated) GIF arrives as
    # ``image/gif`` but we treat it as a frame sequence (video path).
    ext = Path(filename or "").suffix.lower()
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    ct = (content_type or "").lower()
    if ct.startswith("video/"):
        return "video"
    if ct.startswith("image/"):
        return "image"
    if ct.startswith("audio/"):
        return "audio"
    return None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Veritas", description="Deepfake & AI-content detector", version="0.1.0")

    image_detector = ImageDetector(settings.image_model, image_size=settings.image_size)
    audio_detector = AudioDetector(settings.audio_model, sample_rate=settings.audio_sample_rate)
    cache = ResultCache(settings.redis_url, ttl_seconds=settings.cache_ttl_seconds)

    app.state.settings = settings
    app.state.image_detector = image_detector
    app.state.audio_detector = audio_detector
    app.state.cache = cache

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            image_model=image_detector.available,
            audio_model=audio_detector.available,
            cache=cache.backend,
        )

    @app.post("/verify", response_model=Verdict)
    async def verify(file: UploadFile = File(...)) -> Verdict:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="file too large")

        modality = _modality_for(file.filename or "", file.content_type)
        if modality is None:
            raise HTTPException(status_code=415, detail="unsupported media type")

        digest = content_hash(data)
        cached = cache.get(digest)
        if cached is not None:
            cached["cached"] = True
            return Verdict(**cached)

        start = time.perf_counter()
        if modality == "image":
            result = _verify_image(image_detector, data, digest, settings)
        elif modality == "audio":
            result = _verify_audio(audio_detector, data, digest, settings)
        else:
            result = _verify_video(file.filename or "video.mp4", data, digest, settings)
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "verify modality=%s verdict=%s fake_prob=%.4f latency_ms=%.2f",
            modality,
            result["verdict"],
            result["fake_probability"],
            result["latency_ms"],
        )
        cache.set(digest, result)
        return Verdict(**result)

    return app


def _require(detector, modality: str):
    if not detector.available:
        raise HTTPException(status_code=503, detail=f"{modality} model not available; export it first")


def _verify_image(detector: ImageDetector, data: bytes, digest: str, settings: Settings) -> dict:
    _require(detector, "image")
    try:
        fake_prob = detector.fake_probability(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not decode image: {exc}") from exc
    verdict, confidence = verdict_from_probability(fake_prob, settings.fake_threshold)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "modality": "image",
        "model": Path(settings.image_model).name,
        "latency_ms": 0.0,
        "cached": False,
        "content_sha256": digest,
    }


def _verify_audio(detector: AudioDetector, data: bytes, digest: str, settings: Settings) -> dict:
    _require(detector, "audio")
    try:
        fake_prob = detector.fake_probability(data)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not decode audio: {exc}") from exc
    verdict, confidence = verdict_from_probability(fake_prob, settings.fake_threshold)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "fake_probability": fake_prob,
        "modality": "audio",
        "model": Path(settings.audio_model).name,
        "latency_ms": 0.0,
        "cached": False,
        "content_sha256": digest,
    }


def _verify_video(filename: str, data: bytes, digest: str, settings: Settings) -> dict:
    image_model = ImageDetector(settings.image_model, image_size=settings.image_size)
    _require(image_model, "image")

    suffix = Path(filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        from veritas.tasks.app import analyze_video

        # ``.apply`` runs the Celery task synchronously (works without a broker);
        # production can dispatch the same task to a worker via ``.delay``.
        agg = analyze_video.apply(
            args=[tmp.name, settings.image_model],
            kwargs={
                "image_size": settings.image_size,
                "fps": settings.video_fps,
                "max_frames": settings.max_video_frames,
                "threshold": settings.fake_threshold,
            },
        ).get()
    finally:
        os.unlink(tmp.name)

    return {
        "verdict": agg["verdict"],
        "confidence": agg["confidence"],
        "fake_probability": agg["fake_probability"],
        "modality": "video",
        "model": Path(settings.image_model).name,
        "latency_ms": 0.0,
        "cached": False,
        "content_sha256": digest,
        "frames_analyzed": agg["frames_analyzed"],
        "frames": agg["frames"],
    }


# Module-level app for `uvicorn veritas.api.app:app`.
app = create_app()

__all__ = ["create_app", "app"]
