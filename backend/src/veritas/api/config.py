"""Inference-service settings (env-driven via pydantic-settings)."""

from __future__ import annotations

from pydantic import Field
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

    allowed_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    image_size: int = 224
    audio_sample_rate: int = 16000
    fake_threshold: float = Field(default=0.5, ge=0, le=1)
    max_upload_bytes: int = Field(default=26_214_400, gt=0)  # 25 MB

    # Redis cache (optional; falls back to in-memory if unreachable).
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 86_400

    # Video → frames sampling.
    video_fps: float = Field(default=1.0, gt=0)
    max_video_frames: int = Field(default=32, gt=0)


def get_settings() -> Settings:
    return Settings()


__all__ = ["Settings", "get_settings"]
