"""Export fine-tuned ViT / Wav2Vec2 models to ONNX for fast serving.

ONNX Runtime gives lower-latency, dependency-light inference (no torch needed at
serve time). Exports use a thin wrapper that returns only the logits so the
served graph has a clean ``input -> logits`` signature, with a dynamic batch
(and, for audio, time) axis.
"""

from __future__ import annotations

from pathlib import Path

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
    model, output_path: str | Path, *, image_size: int = 224, opset: int = DEFAULT_OPSET
) -> Path:
    """Export a ViT image classifier to ONNX with a dynamic batch axis."""
    import torch

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
    """Load a saved ViT (``save_pretrained`` dir) and export it to ONNX."""
    from transformers import ViTForImageClassification

    model = ViTForImageClassification.from_pretrained(str(model_dir))
    return export_image_onnx(model, output_path, **kwargs)


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
