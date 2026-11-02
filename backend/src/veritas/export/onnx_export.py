"""Export fine-tuned CNN / ViT / Wav2Vec2 models to ONNX for fast serving.

ONNX Runtime gives lower-latency, dependency-light inference (no torch needed at
serve time). Exports use a thin wrapper that returns only the logits so the
served graph has a clean ``input -> logits`` signature, with a dynamic batch
(and, for audio, time) axis.
"""

from __future__ import annotations

import json
from pathlib import Path

from veritas.models.image_model import load_image_classifier
from veritas.models.metadata import (
    default_image_metadata,
    file_sha256,
    metadata_path,
    torch_state_sha256,
)

DEFAULT_OPSET = 17


def _image_wrapper(model):
    import torch

    class ImageLogits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            return self.m(pixel_values=pixel_values).logits

    return ImageLogits(model).eval()


def _audio_wrapper(model):
    import torch

    class AudioLogits(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_values):
            return self.m(input_values=input_values).logits

    return AudioLogits(model).eval()


def export_image_onnx(
    model, output_path: str | Path, *, image_size: int | None = None, opset: int = DEFAULT_OPSET
) -> Path:
    """Export a supported image classifier with its recorded pixel contract."""
    import torch

    declared = getattr(getattr(model, "config", None), "image_size", 224)
    if image_size is not None and image_size != declared:
        raise ValueError(f"Export input size must match checkpoint image_size={declared}")
    image_size = declared
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _image_wrapper(model)
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: stable dynamic_axes support
    )
    metadata = default_image_metadata(image_size)
    if getattr(model.config, "model_type", None) == "veritas_convnext_tiny":
        metadata.update(
            id="Veritas ConvNeXt-Tiny",
            architecture="convnext_tiny",
            source_url="https://github.com/Blasted-ctrl/Veritas/blob/main/docs/cnn-training.md",
            initialization=model.config.backbone_weights or "random",
        )
    metadata["artifact_sha256"] = file_sha256(out)
    metadata["torch_state_sha256"] = torch_state_sha256(model)
    metadata_path(out).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return out


def export_audio_onnx(
    model, output_path: str | Path, *, sample_length: int = 16000, opset: int = DEFAULT_OPSET
) -> Path:
    """Export a Wav2Vec2 classifier to ONNX with dynamic batch + time axes."""
    import torch

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    wrapper = _audio_wrapper(model)
    dummy = torch.randn(1, sample_length)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["input_values"],
        output_names=["logits"],
        dynamic_axes={"input_values": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,  # legacy TorchScript exporter: stable dynamic_axes support
    )
    return out


def export_image_from_dir(model_dir: str | Path, output_path: str | Path, **kwargs) -> Path:
    """Load a supported local image classifier and export its ONNX graph."""
    model = load_image_classifier(model_dir)
    audit = Path(model_dir) / "data-audit.json"
    data = None
    if audit.exists():
        try:
            data = json.loads(audit.read_text(encoding="utf-8"))
            completion = json.loads((Path(model_dir) / "metrics.json").read_text(encoding="utf-8"))["meta"]
            if not isinstance(data["data_kind"], str):
                raise ValueError("Invalid training data classification")
            if not data["fingerprint"] or completion["dataset_fingerprint"] != data["fingerprint"]:
                raise ValueError("Training audit does not match completed training")
            if completion["image_size"] != model.config.image_size:
                raise ValueError("Checkpoint processor size does not match completed training")
            if completion["torch_state_sha256"] != torch_state_sha256(model):
                raise ValueError("Checkpoint weights do not match completed training")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Training provenance requires matching completed metrics and weights") from exc
    path = export_image_onnx(model, output_path, **kwargs)
    if data is not None:
        sidecar = metadata_path(path)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["training_data_fingerprint"] = data["fingerprint"]
        metadata["training_data_kind"] = data["data_kind"]
        sidecar.write_text(json.dumps(metadata, indent=2) + chr(10), encoding="utf-8")
    return path


def export_audio_from_dir(model_dir: str | Path, output_path: str | Path, **kwargs) -> Path:
    """Load a saved Wav2Vec2 (``save_pretrained`` dir) and export it to ONNX."""
    from transformers import Wav2Vec2ForSequenceClassification

    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(model_dir))
    return export_audio_onnx(model, output_path, **kwargs)


__all__ = [
    "export_image_onnx",
    "export_audio_onnx",
    "export_image_from_dir",
    "export_audio_from_dir",
    "DEFAULT_OPSET",
]
