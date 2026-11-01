"""Pydantic response models for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FrameVerdict(BaseModel):
    index: int
    fake_probability: float


class ModelInfo(BaseModel):
    id: str
    source_type: str
    source_url: str | None = None
    revision: str | None = None
    license: str | None = None
    artifact_sha256: str


class Verdict(BaseModel):
    verdict: str = Field(description="'real' or 'fake'")
    confidence: float = Field(
        description="Score for the selected class, not a calibrated probability of authenticity."
    )
    fake_probability: float = Field(description="Model probability that the media is fake, in [0, 1].")
    modality: str = Field(description="'image', 'audio' or 'video'.")
    model: str = Field(description="Backing model identifier.")
    latency_ms: float = Field(description="Server-measured inference latency in milliseconds.")
    cached: bool = Field(default=False, description="Whether the result was served from cache.")
    model_info: ModelInfo
    decision_threshold: float = 0.5
    content_sha256: str
    # Image only: Grad-CAM interpretability overlays as PNG data URLs (present
    # when explanation is requested and an explainer is configured).
    heatmap: str | None = None
    overlay: str | None = None
    # Video only: per-frame breakdown + how the frames were aggregated.
    frames_analyzed: int | None = None
    frames: list[FrameVerdict] | None = None


class HealthResponse(BaseModel):
    status: str
    image_model: bool
    audio_model: bool
    explainer: bool
    cache: str
    max_upload_bytes: int
    video_fps: float
    max_video_frames: int


__all__ = ["Verdict", "FrameVerdict", "HealthResponse"]
