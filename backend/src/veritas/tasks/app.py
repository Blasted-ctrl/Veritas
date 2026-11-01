"""Celery app + the video analysis task.

The task fans a video out into sampled frames, scores each with the ONNX image
detector, and aggregates the per-frame fake-probabilities into a single
verdict. It runs on a distributed worker (``celery -A veritas.tasks.app worker``)
in production, or synchronously via ``analyze_video.apply(...)`` for the
single-process demo and tests.
"""

from __future__ import annotations

import os

from celery import Celery

BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/1")
RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

app = Celery("veritas", broker=BROKER_URL, backend=RESULT_BACKEND)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Eager mode lets the API and tests run the task in-process without a broker.
    task_always_eager=os.environ.get("CELERY_TASK_ALWAYS_EAGER", "0") == "1",
    task_eager_propagates=True,
)


@app.task(name="veritas.analyze_video")
def analyze_video(
    video_path: str,
    model_path: str,
    *,
    image_size: int = 224,
    fps: float = 1.0,
    max_frames: int = 32,
    threshold: float = 0.5,
) -> dict:
    """Score a video by aggregating per-frame fake-probabilities."""
    from veritas.api.inference import ImageDetector
    from veritas.tasks.video import analyze_frames

    return analyze_frames(
        video_path,
        ImageDetector(model_path, image_size=image_size),
        fps=fps,
        max_frames=max_frames,
        threshold=threshold,
    )


__all__ = ["app", "analyze_video"]
