"""Smoke tests for ViT fine-tuning.

Uses a tiny, randomly-initialised ViT (``pretrained=False``) on synthetic data
so the full train -> select -> evaluate -> metrics.json pipeline runs fast on
CPU with no network access. Skipped automatically when the ml extra is absent.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("PIL")

from veritas.data.prepare import prepare  # noqa: E402
from veritas.models.image_model import (  # noqa: E402
    ImageModelConfig,
    build_vit,
    encoder_layer_index,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top_encoder_layers,
)

pytestmark = pytest.mark.ml


def _tiny_config() -> ImageModelConfig:
    return ImageModelConfig(
        pretrained=False,
        image_size=32,
        patch_size=16,
        hidden_size=48,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=96,
    )


def test_freeze_then_unfreeze_changes_trainable_params():
    cfg = _tiny_config()  # 2 encoder blocks
    model = build_vit(cfg)

    freeze_backbone(model)
    head_only, total = trainable_parameter_count(model)
    # Only the classifier head should be trainable.
    assert 0 < head_only < total
    assert all((not p.requires_grad) for n, p in model.named_parameters() if not n.startswith("classifier"))

    unfreeze_top_encoder_layers(model, 1)
    more, _ = trainable_parameter_count(model)

    # Regression guard: unfreezing must touch the actual top encoder block, not
    # merely the final layernorm. The top block carries thousands of weights, so
    # the trainable count must jump well past head + a couple of layernorm vecs.
    assert more > head_only + 5 * cfg.hidden_size

    trainable_top = {
        n
        for n, p in model.named_parameters()
        if p.requires_grad and encoder_layer_index(n) == cfg.num_hidden_layers - 1
    }
    assert trainable_top, "top encoder block was not unfrozen"
    # The bottom block must remain frozen.
    frozen_bottom = {
        n for n, p in model.named_parameters() if (not p.requires_grad) and encoder_layer_index(n) == 0
    }
    assert frozen_bottom, "bottom encoder block should stay frozen"


def test_train_image_smoke_produces_metrics(tmp_path):
    from veritas.training.train_image import TrainImageConfig, train_image

    data_dir = tmp_path / "data"
    prepare(
        modality="image",
        source="synthetic",
        output_dir=data_dir,
        synthetic_subjects=16,
        synthetic_per_subject=4,
        seed=1337,
    )

    out_dir = tmp_path / "model"
    cfg = TrainImageConfig(
        data_dir=data_dir,
        output_dir=out_dir,
        pretrained=False,
        image_size=32,
        epochs=2,
        batch_size=8,
        freeze_epochs=1,
        unfreeze_layers=1,
        seed=1337,
        device="cpu",
    )
    # Shrink the model the trainer builds to the tiny config dims.
    cfg.model_name = "vit-tiny-random"

    metrics = train_image(cfg)

    # metrics.json written and well-formed.
    doc = json.loads((out_dir / "metrics.json").read_text())
    assert set(["accuracy", "precision", "recall", "f1", "auc"]).issubset(doc)
    assert doc["meta"]["modality"] == "image"
    assert doc["meta"]["test_size"] > 0
    assert 0.0 <= metrics.accuracy <= 1.0
    # Model artifacts saved for ONNX export later.
    assert (out_dir / "config.json").exists()


def test_train_image_is_deterministic(tmp_path):
    from veritas.training.train_image import TrainImageConfig, train_image

    data_dir = tmp_path / "data"
    prepare(modality="image", output_dir=data_dir, synthetic_subjects=16, seed=1337)

    def run(sub):
        cfg = TrainImageConfig(
            data_dir=data_dir,
            output_dir=tmp_path / sub,
            pretrained=False,
            image_size=32,
            epochs=1,
            batch_size=8,
            freeze_epochs=0,
            unfreeze_layers=0,
            seed=7,
            device="cpu",
        )
        return train_image(cfg)

    a = run("a")
    b = run("b")
    assert a.accuracy == b.accuracy
    assert a.auc == b.auc
