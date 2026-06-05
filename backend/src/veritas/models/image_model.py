"""Vision Transformer (ViT) image manipulation detector.

Wraps a Hugging Face ``ViTForImageClassification`` for binary real/fake
classification and provides the freezing/unfreezing strategy used during
fine-tuning. Also exposes a deterministic, network-free preprocessing transform
so training, inference and tests all agree on pixel normalization.

Requires the optional ``ml`` extra (torch/torchvision/transformers).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches a ViT encoder block index across transformers naming schemes:
#   transformers >=5.x : "vit.layers.<i>.*"
#   transformers <5.x  : "vit.encoder.layer.<i>.*"
_ENCODER_LAYER_RE = re.compile(r"(?:encoder\.layer|\.layers)\.(\d+)\.")

# ImageNet statistics — ViT checkpoints are pretrained with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_MODEL = "google/vit-base-patch16-224"
NUM_LABELS = 2  # 0 = REAL, 1 = FAKE


@dataclass
class ImageModelConfig:
    pretrained_name: str = DEFAULT_MODEL
    pretrained: bool = True
    image_size: int = 224
    patch_size: int = 16
    # Used only when pretrained=False (tiny randomly-initialised model for
    # fast, network-free smoke training/tests).
    hidden_size: int = 192
    num_hidden_layers: int = 4
    num_attention_heads: int = 3
    intermediate_size: int = 768


def make_image_transform(image_size: int = 224):
    """A deterministic eval/inference transform: resize -> tensor -> normalize."""
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_vit(config: ImageModelConfig):
    """Construct a ``ViTForImageClassification`` for binary classification."""
    from transformers import ViTConfig, ViTForImageClassification

    id2label = {0: "real", 1: "fake"}
    label2id = {"real": 0, "fake": 1}

    if config.pretrained:
        model = ViTForImageClassification.from_pretrained(
            config.pretrained_name,
            num_labels=NUM_LABELS,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else:
        vit_config = ViTConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=3,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            num_labels=NUM_LABELS,
            id2label=id2label,
            label2id=label2id,
        )
        model = ViTForImageClassification(vit_config)
    return model


def freeze_backbone(model) -> None:
    """Freeze the ViT encoder/embeddings; train only the classification head.

    This is stage 1 of the strategy: cheaply adapt the head to the new task
    before risking the pretrained features.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def encoder_layer_index(param_name: str) -> int | None:
    """Return the encoder-block index encoded in ``param_name``, else ``None``.

    Per-block sublayers (attention, MLP, the block's own layernorms) all carry
    the block index; the final encoder-output ``vit.layernorm`` does not.
    """
    match = _ENCODER_LAYER_RE.search(param_name)
    return int(match.group(1)) if match else None


def unfreeze_top_encoder_layers(model, n_layers: int) -> None:
    """Unfreeze the top ``n_layers`` transformer blocks (plus head + final norm).

    Stage 2: gradually un-freeze the most task-specific (top) layers so the
    backbone can specialise to manipulation artefacts without catastrophically
    forgetting its pretrained representations. Embeddings and the lower blocks
    stay frozen.
    """
    total = model.config.num_hidden_layers
    keep_frozen_below = max(0, total - n_layers)
    for name, param in model.named_parameters():
        idx = encoder_layer_index(name)
        if name.startswith("classifier"):
            param.requires_grad = True
        elif idx is not None:
            param.requires_grad = idx >= keep_frozen_below
        elif "layernorm" in name:  # final encoder-output layernorm (no block idx)
            param.requires_grad = True
        else:  # embeddings / pooler stay frozen
            param.requires_grad = False


def trainable_parameter_count(model) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


__all__ = [
    "ImageModelConfig",
    "build_vit",
    "make_image_transform",
    "freeze_backbone",
    "unfreeze_top_encoder_layers",
    "trainable_parameter_count",
    "DEFAULT_MODEL",
    "NUM_LABELS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
]
