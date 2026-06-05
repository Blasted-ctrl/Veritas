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

    reader = imageio.get_reader(str(video_path))
    try:
        meta = reader.get_meta_data()
        native_fps = float(meta.get("fps") or 0.0)
    except Exception:
        native_fps = 0.0

    # Stride between kept frames. If the native fps is unknown (e.g. GIF), keep
    # every frame up to the cap.
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
    reader.close()
    return pngs


__all__ = ["extract_frame_pngs"]
