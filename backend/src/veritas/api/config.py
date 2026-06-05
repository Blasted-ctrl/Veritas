"""Inference-service settings (env-driven via pydantic-settings)."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from veritas.config import PATHS


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERITAS_", extra="ignore")

    # Exported ONNX model paths.
    image_model: str = str(PATHS.models / "image" / "model.onnx")
    audio_model: str = str(PATHS.models / "audio" / "model.onnx")

    # Optional saved torch ViT directory used for Grad-CAM heatmaps (needs the
    # ``ml`` extra). When unset/absent, /verify still works but without heatmaps.
    image_torch_dir: str = str(PATHS.models / "image")

    image_size: int = 224
    audio_sample_rate: int = 16000
    fake_threshold: float = 0.5
    max_upload_bytes: int = 26_214_400  # 25 MB

    # Redis cache (optional; falls back to in-memory if unreachable).
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86_400

    # Video → frames sampling.
    video_fps: float = 1.0
    max_video_frames: int = 32


def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
