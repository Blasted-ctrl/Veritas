"""Smoke tests for Wav2Vec2 fine-tuning.

Uses a tiny, randomly-initialised Wav2Vec2 (``pretrained=False``) on synthetic
audio so the full pipeline runs fast on CPU with no network access. Skipped
automatically when the ml extra is absent.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("soundfile")

from veritas.data.prepare import prepare  # noqa: E402
from veritas.models.audio_model import (  # noqa: E402
    AudioModelConfig,
    build_wav2vec2,
    encoder_layer_index,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top_encoder_layers,
)

pytestmark = pytest.mark.ml


def _tiny_config() -> AudioModelConfig:
    return AudioModelConfig(
        pretrained=False,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )


def test_freeze_then_unfreeze_changes_trainable_params():
    cfg = _tiny_config()
    model = build_wav2vec2(cfg)

    freeze_backbone(model)
    head_only, total = trainable_parameter_count(model)
    assert 0 < head_only < total
    # Only the projector + classifier head should be trainable.
    assert all(
        (not p.requires_grad)
        for n, p in model.named_parameters()
        if not (n.startswith("projector") or n.startswith("classifier"))
    )

    unfreeze_top_encoder_layers(model, 1)
    more, _ = trainable_parameter_count(model)
    assert more > head_only + 5 * cfg.hidden_size  # real encoder block, not just norms

    trainable_top = {
        n
        for n, p in model.named_parameters()
        if p.requires_grad and encoder_layer_index(n) == cfg.num_hidden_layers - 1
    }
    assert trainable_top, "top encoder block was not unfrozen"
    frozen_bottom = {
        n for n, p in model.named_parameters() if (not p.requires_grad) and encoder_layer_index(n) == 0
    }
    assert frozen_bottom, "bottom encoder block should stay frozen"


def test_train_audio_smoke_produces_metrics(tmp_path):
    from veritas.training.train_audio import TrainAudioConfig, train_audio

    data_dir = tmp_path / "data"
    prepare(
        modality="audio",
        source="synthetic",
        output_dir=data_dir,
        synthetic_subjects=16,
        synthetic_per_subject=4,
        seed=1337,
    )

    out_dir = tmp_path / "model"
    cfg = TrainAudioConfig(
        data_dir=data_dir,
        output_dir=out_dir,
        pretrained=False,
        max_seconds=0.5,
        epochs=2,
        batch_size=8,
        freeze_epochs=1,
        unfreeze_layers=1,
        seed=1337,
        device="cpu",
    )
    metrics = train_audio(cfg)

    doc = json.loads((out_dir / "metrics.json").read_text())
    assert {"accuracy", "precision", "recall", "f1", "auc"}.issubset(doc)
    assert doc["meta"]["modality"] == "audio"
    assert doc["meta"]["test_size"] > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    assert (out_dir / "config.json").exists()
