"""Pydantic response models for the inference API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FrameVerdict(BaseModel):
    index: int
    fake_probability: float


class Verdict(BaseModel):
    verdict: str = Field(description="'real' or 'fake'")
    confidence: float = Field(description="Confidence in the reported verdict, in [0.5, 1].")
    fake_probability: float = Field(description="Model probability that the media is fake, in [0, 1].")
    modality: str = Field(description="'image', 'audio' or 'video'.")
    model: str = Field(description="Backing model identifier.")
    latency_ms: float = Field(description="Server-measured inference latency in milliseconds.")
    cached: bool = Field(default=False, description="Whether the result was served from cache.")
    content_sha256: str
    # Video only: per-frame breakdown + how the frames were aggregated.
    frames_analyzed: int | None = None
    frames: list[FrameVerdict] | None = None


class HealthResponse(BaseModel):
    status: str
    image_model: bool
    audio_model: bool
    cache: str


__all__ = ["Verdict", "FrameVerdict", "HealthResponse"]
