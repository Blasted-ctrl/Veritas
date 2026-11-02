"""ConvNeXt-Tiny binary classifier with a local, weights-only checkpoint format."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

from veritas.models.metadata import file_sha256


@dataclass(frozen=True)
class CNNConfig:
    image_size: int = 224
    backbone_weights: str | None = None
    model_type: str = "veritas_convnext_tiny"
    num_labels: int = 2


class ImageOutput(NamedTuple):
    logits: torch.Tensor
    loss: torch.Tensor | None


class ConvNeXtDetector(nn.Module):
    def __init__(self, *, pretrained: bool = True, image_size: int = 224):
        super().__init__()
        if not isinstance(image_size, int) or not 32 <= image_size <= 4096:
            raise ValueError("ConvNeXt image_size must be between 32 and 4096")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        network = convnext_tiny(weights=weights)
        network.classifier[2] = nn.Linear(network.classifier[2].in_features, 2)
        self.features = network.features
        self.avgpool = network.avgpool
        self.classifier = network.classifier
        self.config = CNNConfig(image_size, weights.name if weights else None)

    def forward(self, pixel_values, labels=None):
        logits = self.classifier(self.avgpool(self.features(pixel_values)))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return ImageOutput(logits, loss)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep fixed features deterministic while only the head is learning.
        if mode and not any(parameter.requires_grad for parameter in self.features.parameters()):
            self.features.eval()
        return self

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".cnn-", suffix=".pt", dir=directory)
        os.close(handle)
        try:
            torch.save({key: value.detach().cpu() for key, value in self.state_dict().items()}, temporary)
            digest = file_sha256(temporary)
            os.replace(temporary, directory / "model.pt")
            config = asdict(self.config) | {"checkpoint_sha256": digest}
            handle, metadata = tempfile.mkstemp(prefix=".cnn-", suffix=".json", dir=directory)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    json.dump(config, stream, indent=2)
                    stream.write("\n")
                os.replace(metadata, directory / "config.json")
            finally:
                Path(metadata).unlink(missing_ok=True)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def from_pretrained(cls, directory: str | Path):
        directory = Path(directory)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "veritas_convnext_tiny" or config.get("num_labels") != 2:
            raise ValueError("Unsupported CNN checkpoint configuration")
        if config.get("backbone_weights") not in (None, "IMAGENET1K_V1"):
            raise ValueError("Unsupported CNN initialization provenance")
        weights = directory / "model.pt"
        if file_sha256(weights) != config.get("checkpoint_sha256"):
            raise ValueError("CNN checkpoint checksum does not match its configuration")
        model = cls(pretrained=False, image_size=config["image_size"])
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
        model.config = CNNConfig(config["image_size"], config["backbone_weights"])
        return model


def unfreeze_top_stages(model: ConvNeXtDetector, n_stages: int) -> None:
    if not 0 <= n_stages <= 4:
        raise ValueError("ConvNeXt has four stages; choose a value from 0 to 4")
    first = (4 - n_stages) * 2
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier") or (
            name.startswith("features.") and int(name.split(".")[1]) >= first
        )
