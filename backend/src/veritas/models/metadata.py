"""The processor and provenance contract stored beside an image ONNX graph."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from veritas.models.image_model import IMAGENET_MEAN, IMAGENET_STD


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path(path: str | Path) -> Path:
    return Path(str(path) + ".json")


def default_image_metadata(image_size: int = 224) -> dict:
    return {
        "format_version": 1,
        "id": "Custom image model",
        "source_type": "custom",
        "image_size": image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "resample": 2,
        "rescale_factor": 1 / 255,
        "fake_index": 1,
    }


def load_image_metadata(path: str | Path, *, image_size: int = 224) -> dict:
    info = default_image_metadata(image_size)
    sidecar = metadata_path(path)
    if sidecar.exists():
        contents = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(contents, dict):
            raise ValueError("model metadata must be an object")
        info.update(contents)
    if info["format_version"] != 1:
        raise ValueError("unsupported model metadata version")
    if not isinstance(info["image_size"], int) or not 1 <= info["image_size"] <= 4096:
        raise ValueError("invalid model input size")
    for key in ("mean", "std"):
        values = info[key]
        if not isinstance(values, list) or len(values) != 3:
            raise ValueError(f"{key} must contain three channel values")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError(f"invalid {key}")
    if any(v <= 0 for v in info["std"]):
        raise ValueError("channel standard deviations must be positive")
    if info["fake_index"] not in (0, 1) or info["resample"] not in range(6):
        raise ValueError("invalid class mapping or resampling filter")
    scale = info["rescale_factor"]
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid pixel rescaling factor")
    expected = info.get("artifact_sha256")
    if expected and file_sha256(path) != expected:
        raise ValueError("model checksum does not match its metadata")
    return info


def torch_state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
