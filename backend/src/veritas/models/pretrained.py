"""Explicit, checksum-verified installation of the public pretrained detector."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

from veritas.models.metadata import file_sha256, metadata_path

MODEL_REPO = "onnx-community/ai-image-detection-ONNX"
MODEL_REVISION = "e3cfe99f2841930a040a6281682c10c989965603"
MODEL_SHA256 = "44cb205f596f2530686bf4895e3678f08f9483c926b6bdc0e02998d5b54978a2"
MODEL_SIZE = 343_401_688
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/onnx/model.onnx"


def download_model(output: str | Path, *, overwrite: bool = False) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    installed = output.exists() and file_sha256(output) == MODEL_SHA256
    if output.exists() and not installed and not overwrite:
        raise FileExistsError("A different model already exists. Choose another output or use --overwrite.")
    if not installed:
        if shutil.disk_usage(output.parent).free < MODEL_SIZE + 10_485_760:
            raise OSError("Model installation needs at least 338 MiB of free working space.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".partial", delete=False) as stream:
                temporary = Path(stream.name)
                total = 0
                with urlopen(MODEL_URL, timeout=60) as source:
                    while chunk := source.read(1_048_576):
                        total += len(chunk)
                        if total > MODEL_SIZE:
                            raise ValueError("Downloaded model exceeds its expected size")
                        stream.write(chunk)
            if total != MODEL_SIZE or file_sha256(temporary) != MODEL_SHA256:
                raise ValueError("Downloaded model failed its size/checksum verification")
            os.replace(temporary, output)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    info = {
        "format_version": 1,
        "id": MODEL_REPO,
        "source_type": "pretrained",
        "source_url": f"https://huggingface.co/{MODEL_REPO}",
        "revision": MODEL_REVISION,
        "license": "Apache-2.0",
        "artifact_sha256": MODEL_SHA256,
        "upstream": "capcheck/ai-image-detection (dima806 CIFAKE ViT)",
        "image_size": 224,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "resample": 2,
        "rescale_factor": 1 / 255,
        "fake_index": 1,
    }
    metadata_path(output).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return output
