"""Wav2Vec2 synthetic-voice (voice-clone) detector.

Wraps a Hugging Face ``Wav2Vec2ForSequenceClassification`` for binary
bona-fide/spoof classification and provides audio loading + resampling and the
freezing strategy used during fine-tuning.

Requires the optional ``ml`` extra (torch/torchaudio/transformers/soundfile).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

TARGET_SAMPLE_RATE = 16000  # Wav2Vec2 is pretrained on 16 kHz audio
DEFAULT_MODEL = "facebook/wav2vec2-base"
NUM_LABELS = 2  # 0 = REAL/bona-fide, 1 = FAKE/spoof

# Encoder block index across transformers naming schemes.
_ENCODER_LAYER_RE = re.compile(r"encoder\.layers\.(\d+)\.")


@dataclass
class AudioModelConfig:
    pretrained_name: str = DEFAULT_MODEL
    pretrained: bool = True
    # Used only when pretrained=False (tiny randomly-initialised model for fast,
    # network-free smoke training/tests).
    hidden_size: int = 32
    num_hidden_layers: int = 2
    num_attention_heads: int = 2
    intermediate_size: int = 64


def load_audio(path: str, target_sr: int = TARGET_SAMPLE_RATE):
    """Load an audio file to a mono 1-D float32 tensor at ``target_sr``."""
    import soundfile as sf
    import torch

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).mean(dim=1)  # mix down to mono
    if sr != target_sr:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform


def build_wav2vec2(config: AudioModelConfig):
    """Construct a ``Wav2Vec2ForSequenceClassification`` for binary classification."""
    from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

    id2label = {0: "real", 1: "fake"}
    label2id = {"real": 0, "fake": 1}

    if config.pretrained:
        return Wav2Vec2ForSequenceClassification.from_pretrained(
            config.pretrained_name,
            num_labels=NUM_LABELS,
            id2label=id2label,
            label2id=label2id,
        )

    # A tiny config whose convolutional feature extractor still consumes raw
    # 16 kHz audio but with far fewer parameters (for smoke tests).
    w2v_config = Wav2Vec2Config(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        conv_dim=(32, 32, 32),
        conv_stride=(5, 2, 2),
        conv_kernel=(10, 3, 3),
        num_feat_extract_layers=3,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        classifier_proj_size=config.hidden_size,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )
    return Wav2Vec2ForSequenceClassification(w2v_config)


def encoder_layer_index(param_name: str) -> int | None:
    """Return the transformer encoder-block index in ``param_name``, else None."""
    match = _ENCODER_LAYER_RE.search(param_name)
    return int(match.group(1)) if match else None


def _is_head(name: str) -> bool:
    # The sequence-classification head: projector + classifier (+ pooling).
    return name.startswith("projector") or name.startswith("classifier")


def freeze_backbone(model) -> None:
    """Stage 1: freeze the CNN feature extractor + transformer; train the head."""
    # The convolutional feature encoder is always frozen during fine-tuning
    # (standard Wav2Vec2 practice — it learns generic acoustic features).
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    for name, param in model.named_parameters():
        param.requires_grad = _is_head(name)


def unfreeze_top_encoder_layers(model, n_layers: int) -> None:
    """Stage 2: unfreeze the top ``n_layers`` transformer blocks (+ head)."""
    total = model.config.num_hidden_layers
    keep_frozen_below = max(0, total - n_layers)
    for name, param in model.named_parameters():
        idx = encoder_layer_index(name)
        if _is_head(name):
            param.requires_grad = True
        elif idx is not None:
            param.requires_grad = idx >= keep_frozen_below
        elif name.startswith("wav2vec2.encoder.layer_norm") or name.startswith(
            "wav2vec2.encoder.pos_conv_embed"
        ):
            param.requires_grad = True
        else:  # feature_extractor / feature_projection stay frozen
            param.requires_grad = False
    # Keep the conv feature encoder frozen regardless.
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()


def trainable_parameter_count(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


__all__ = [
    "AudioModelConfig",
    "build_wav2vec2",
    "load_audio",
    "freeze_backbone",
    "unfreeze_top_encoder_layers",
    "encoder_layer_index",
    "trainable_parameter_count",
    "TARGET_SAMPLE_RATE",
    "DEFAULT_MODEL",
    "NUM_LABELS",
]
