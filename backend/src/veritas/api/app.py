"""FastAPI inference service: POST /verify returns an authenticity verdict.

Routing by modality:
  * image -> ONNX CNN/ViT detector
  * audio -> ONNX Wav2Vec2 detector
  * video -> sampled image frames, averaged in the API process

Results are cached by content hash; inference latency is measured and returned
(and logged). The app degrades gracefully: missing models yield 503, an
unreachable Redis falls back to an in-memory cache.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from veritas.api.cache import ResultCache, content_hash
from veritas.api.config import Settings, get_settings
from veritas.api.explainer import GradCamExplainer
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

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    image_detector = ImageDetector(settings.image_model, image_size=settings.image_size)
    audio_detector = AudioDetector(settings.audio_model, sample_rate=settings.audio_sample_rate)
    cache = ResultCache(settings.redis_url, ttl_seconds=settings.cache_ttl_seconds)
    explainer = GradCamExplainer(settings.image_torch_dir, image_size=settings.image_size)

    app.state.settings = settings
    app.state.image_detector = image_detector
    app.state.audio_detector = audio_detector
    app.state.cache = cache
    app.state.explainer = explainer

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            image_model=image_detector.available,
            audio_model=audio_detector.available,
            explainer=explainer.matches(image_detector.metadata) if image_detector.available else False,
            cache=cache.backend,
            max_upload_bytes=settings.max_upload_bytes,
            video_fps=settings.video_fps,
            max_video_frames=settings.max_video_frames,
        )

    @app.post("/verify", response_model=Verdict, response_model_exclude_none=True)
    async def verify(file: UploadFile = File(...), explain: bool = False) -> Verdict:
        data = await file.read(settings.max_upload_bytes + 1)
        if not data:
            raise HTTPException(status_code=400, detail="empty upload")
        if len(data) > settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="file too large")

        modality = _modality_for(file.filename or "", file.content_type)
        if modality is None:
            raise HTTPException(status_code=415, detail="unsupported media type")

        detector = audio_detector if modality == "audio" else image_detector
        _require(detector, modality)
        digest = content_hash(data)
        identity = {
            "content": digest,
            "modality": modality,
            "model": detector.fingerprint,
            "threshold": settings.fake_threshold,
            "fps": settings.video_fps if modality == "video" else None,
            "max_frames": settings.max_video_frames if modality == "video" else None,
            "explain": explain,
            "explainer": explainer.model_dir.as_posix() if explain else None,
        }
        cache_key = content_hash(json.dumps(identity, sort_keys=True).encode())
        cached = cache.get(cache_key)
        if cached is not None and not (
            explain
            and modality == "image"
            and not cached.get("overlay")
            and explainer.matches(image_detector.metadata)
        ):
            cached["cached"] = True
            return Verdict(**cached)

        start = time.perf_counter()
        if modality == "image":
            result = await run_in_threadpool(_verify_image, image_detector, data, digest, settings)
            if explain:
                await run_in_threadpool(_attach_heatmap, explainer, image_detector, data, result)
        elif modality == "audio":
            result = await run_in_threadpool(_verify_audio, audio_detector, data, digest, settings)
        else:
            result = await run_in_threadpool(
                _verify_video, file.filename or "video.mp4", data, digest, settings, image_detector
            )
        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)

        logger.info(
            "verify modality=%s verdict=%s fake_prob=%.4f latency_ms=%.2f",
            modality,
            result["verdict"],
            result["fake_probability"],
            result["latency_ms"],
        )
        result["model_info"] = detector.model_info
        result["decision_threshold"] = settings.fake_threshold
        cache.set(cache_key, result)
        return Verdict(**result)

    return app


def _require(detector, modality: str):
    if not detector.available:
        raise HTTPException(
            status_code=503, detail=f"{modality} model is not available; install or export a compatible model"
        )


def _attach_heatmap(explainer: GradCamExplainer, detector: ImageDetector, data: bytes, result: dict) -> None:
    """Best-effort Grad-CAM overlay; never fails the verdict if explanation errors."""
    if not explainer.available:
        return
    try:
        explanation = explainer.explain(
            data, class_idx=detector.metadata["fake_index"], preprocessing=detector.metadata
        )
    except Exception:  # interpretability is optional — don't break /verify
        explanation = None
    if explanation:
        result["heatmap"] = explanation["heatmap"]
        result["overlay"] = explanation["overlay"]


def _verify_image(detector: ImageDetector, data: bytes, digest: str, settings: Settings) -> dict:
    _require(detector, "image")
    try:
        fake_prob = detector.fake_probability(data)
    except Exception as exc:
        logger.info("Image analysis failed: %s", exc)
        raise HTTPException(
            status_code=422, detail="Could not decode this image. Choose a readable image file."
        ) from exc
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
        logger.info("Audio analysis failed: %s", exc)
        raise HTTPException(
            status_code=422, detail="Could not decode this audio. Choose a readable audio file."
        ) from exc
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


def _verify_video(
    filename: str, data: bytes, digest: str, settings: Settings, detector: ImageDetector
) -> dict:
    from veritas.tasks.video import analyze_frames

    suffix = Path(filename).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        try:
            agg = analyze_frames(
                tmp.name,
                detector,
                fps=settings.video_fps,
                max_frames=settings.max_video_frames,
                threshold=settings.fake_threshold,
            )
        except Exception as exc:
            logger.info("Video decoding failed: %s", exc)
            raise HTTPException(status_code=422, detail="Could not decode readable video frames") from exc
    finally:
        tmp.close()
        os.unlink(tmp.name)
    return {
        **agg,
        "modality": "video",
        "model": Path(settings.image_model).name,
        "latency_ms": 0.0,
        "cached": False,
        "content_sha256": digest,
    }


# Module-level app for `uvicorn veritas.api.app:app`.
app = create_app()

__all__ = ["create_app", "app"]
