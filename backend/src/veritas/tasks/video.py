"""Video frame extraction for the Celery analysis pipeline.

Uses ``imageio`` so common containers (mp4 via imageio-ffmpeg, gif via Pillow)
are supported uniformly. Frames are sampled at a target FPS and capped, then
re-encoded to PNG bytes so the rest of the pipeline reuses the image path.
"""

from __future__ import annotations

import io
from pathlib import Path


def extract_frame_pngs(video_path: str | Path, *, fps: float = 1.0, max_frames: int = 32) -> list[bytes]:
    """Sample frames from a video and return them as PNG byte strings."""
    import imageio.v2 as imageio
    from PIL import Image

    format_name = "GIF-PIL" if Path(video_path).suffix.lower() == ".gif" else "FFMPEG"
    # Own the request before constructing its legacy reader. A corrupt file can
    # fail during reader construction; imopen still releases the file handle.
    with imageio.imopen(str(video_path), "r", plugin=format_name, legacy_mode=True) as resource:
        reader = resource.legacy_get_reader()
        try:
            try:
                meta = reader.get_meta_data()
                native_fps = float(meta.get("fps") or 0.0)
            except Exception:
                native_fps = 0.0
            stride = max(1, int(round(native_fps / fps))) if native_fps and fps > 0 else 1
            pngs: list[bytes] = []
            for i, frame in enumerate(reader):
                if i % stride != 0:
                    continue
                image = Image.fromarray(frame).convert("RGB")
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                pngs.append(buf.getvalue())
                if len(pngs) >= max_frames:
                    break
            return pngs
        finally:
            reader.close()


def analyze_frames(video_path, detector, *, fps=1.0, max_frames=32, threshold=0.5) -> dict:
    """Score sampled frames using the caller's already loaded image detector."""
    from veritas.api.inference import verdict_from_probability

    frames = extract_frame_pngs(video_path, fps=fps, max_frames=max_frames)
    if not frames:
        raise ValueError("No readable video frames were found")
    per_frame = [
        {"index": i, "fake_probability": detector.fake_probability(png)} for i, png in enumerate(frames)
    ]
    mean_fake = sum(frame["fake_probability"] for frame in per_frame) / len(per_frame)
    verdict, confidence = verdict_from_probability(mean_fake, threshold)
    return {
        "verdict": verdict,
        "confidence": confidence,
        "fake_probability": mean_fake,
        "frames_analyzed": len(per_frame),
        "frames": per_frame,
    }
