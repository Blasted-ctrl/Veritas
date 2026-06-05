"""Fine-tune a ViT image manipulation detector.

Strategy:
  * deterministic seeding (data, weights init, dataloader shuffling);
  * stage 1 — freeze the pretrained backbone, train only the head;
  * stage 2 — unfreeze the top encoder layers and fine-tune end-to-end;
  * discriminative learning rates (higher for the head, lower for the backbone)
    with linear warmup + decay;
  * model selection on the validation split; final, honest evaluation on the
    held-out test split written to ``metrics.json``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

from veritas.models.image_model import (
    DEFAULT_MODEL,
    ImageModelConfig,
    build_vit,
    freeze_backbone,
    trainable_parameter_count,
    unfreeze_top_encoder_layers,
)
from veritas.training.image_data import ImageManifestDataset, collate
from veritas.training.metrics import ClassificationMetrics, compute_metrics, write_metrics
from veritas.training.seed import seed_everything, seed_worker


@dataclass
class TrainImageConfig:
    data_dir: Path
    output_dir: Path
    model_name: str = DEFAULT_MODEL
    pretrained: bool = True
    image_size: int = 224
    epochs: int = 5
    batch_size: int = 16
    backbone_lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    freeze_epochs: int = 1
    unfreeze_layers: int = 4
    seed: int = 1337
    num_workers: int = 0
    limit: int | None = None
    device: str | None = None


def _resolve_device(requested: str | None):
    import torch

    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(dataset, *, batch_size, shuffle, seed, num_workers):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def _param_groups(model, cfg: TrainImageConfig):
    head, backbone = [], []
    for name, param in model.named_parameters():
        (head if name.startswith("classifier") else backbone).append(param)
    return [
        {"params": head, "lr": cfg.head_lr},
        {"params": backbone, "lr": cfg.backbone_lr},
    ]


def evaluate(model, loader, device) -> tuple[list[int], list[float]]:
    """Return ``(labels, fake_probabilities)`` for every item in ``loader``."""
    import torch

    model.eval()
    labels: list[int] = []
    probs: list[float] = []
    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values).logits
            p = torch.softmax(logits, dim=-1)[:, 1]
            probs.extend(p.detach().cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return labels, probs


def _train_one_epoch(model, loader, optimizer, scheduler, device) -> float:
    import torch

    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        targets = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(pixel_values=pixel_values, labels=targets)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += float(out.loss.detach()) * len(targets)
        n += len(targets)
    return total_loss / max(1, n)


def train_image(cfg: TrainImageConfig) -> ClassificationMetrics:
    """Run the full fine-tune + evaluation pipeline; returns test metrics."""
    import torch
    from transformers import get_linear_schedule_with_warmup

    seed_everything(cfg.seed)
    device = _resolve_device(cfg.device)
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ImageManifestDataset(data_dir / "train.csv", image_size=cfg.image_size, limit=cfg.limit)
    val_ds = ImageManifestDataset(data_dir / "val.csv", image_size=cfg.image_size, limit=cfg.limit)
    test_ds = ImageManifestDataset(data_dir / "test.csv", image_size=cfg.image_size, limit=cfg.limit)

    train_loader = _make_loader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed, num_workers=cfg.num_workers
    )
    val_loader = _make_loader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed, num_workers=cfg.num_workers
    )
    test_loader = _make_loader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, seed=cfg.seed, num_workers=cfg.num_workers
    )

    model = build_vit(
        ImageModelConfig(
            pretrained_name=cfg.model_name,
            pretrained=cfg.pretrained,
            image_size=cfg.image_size,
        )
    ).to(device)

    # Stage 1: head only.
    freeze_backbone(model)

    optimizer = torch.optim.AdamW(_param_groups(model, cfg), weight_decay=cfg.weight_decay)
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    best_auc = -1.0
    best_state = None
    history = []
    start = time.time()

    for epoch in range(cfg.epochs):
        # Stage 2: unfreeze the top encoder layers once the head has adapted.
        if epoch == cfg.freeze_epochs and cfg.unfreeze_layers > 0:
            unfreeze_top_encoder_layers(model, cfg.unfreeze_layers)

        train_loss = _train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_labels, val_probs = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, val_probs)
        trainable, total = trainable_parameter_count(model)
        # Select on AUC when defined, else on F1 (tiny/degenerate val sets).
        score = val_metrics.auc if val_metrics.auc is not None else val_metrics.f1
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_accuracy": val_metrics.accuracy,
                "val_f1": val_metrics.f1,
                "val_auc": val_metrics.auc,
                "trainable_params": trainable,
            }
        )
        print(
            f"epoch {epoch}: loss={train_loss:.4f} "
            f"val_acc={val_metrics.accuracy} val_f1={val_metrics.f1} val_auc={val_metrics.auc} "
            f"trainable={trainable}/{total}"
        )
        if score is not None and score > best_auc:
            best_auc = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final, honest evaluation on the held-out test split.
    test_labels, test_probs = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_labels, test_probs)

    model.save_pretrained(out_dir)
    write_metrics(
        test_metrics,
        out_dir / "metrics.json",
        extra={
            "modality": "image",
            "model_name": cfg.model_name if cfg.pretrained else "vit-tiny-random",
            "pretrained": cfg.pretrained,
            "epochs": cfg.epochs,
            "seed": cfg.seed,
            "device": str(device),
            "image_size": cfg.image_size,
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "test_size": len(test_ds),
            "wall_time_sec": round(time.time() - start, 2),
            "history": history,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        },
    )
    print(
        f"\nTEST  acc={test_metrics.accuracy} precision={test_metrics.precision} "
        f"recall={test_metrics.recall} f1={test_metrics.f1} auc={test_metrics.auc}"
    )
    print(f"Saved model + metrics.json -> {out_dir}")
    return test_metrics


__all__ = ["TrainImageConfig", "train_image", "evaluate"]
