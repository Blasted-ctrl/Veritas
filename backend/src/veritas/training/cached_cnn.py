"""CPU-feasible CNN fine-tuning with an immutable, checksum-bound prefix cache."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from veritas.data.image_corpus import load_image_samples
from veritas.models.metadata import file_sha256, torch_state_sha256
from veritas.training.image_data import ImageManifestDataset
from veritas.training.metrics import compute_metrics, write_metrics
from veritas.training.seed import seed_everything


@dataclass(frozen=True)
class FeatureCache:
    path: Path
    key: str
    labels: tuple[int, ...]
    views: int


@dataclass(frozen=True)
class CachedTrainConfig:
    epochs: int = 8
    batch_size: int = 16
    backbone_lr: float = 2e-5
    head_lr: float = 2e-4
    real_loss_weight: float = 1.0
    weight_decay: float = 0.01
    freeze_epochs: int = 1
    seed: int = 20261101

    def __post_init__(self):
        if self.epochs < 1 or self.batch_size < 1 or self.freeze_epochs < 0:
            raise ValueError("Invalid epoch, batch, or freezing configuration")
        if (
            any(
                not math.isfinite(value) or value <= 0
                for value in (self.backbone_lr, self.head_lr, self.real_loss_weight)
            )
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid learning rate, class weight or decay")


def ensure_feature_cache(model, manifest, cache_dir, *, split, views=1, seed=20261101) -> FeatureCache:
    import numpy as np
    import PIL
    import torch
    import torchvision
    from torch.utils.data import DataLoader

    import veritas.models.image_model as image_processor
    import veritas.training.image_data as image_data

    if split not in ("train", "val") or views < 1 or (split == "val" and views != 1):
        raise ValueError("Fitting caches allow only train/val split roles")
    samples = load_image_samples(manifest)
    if not samples or any(sample.split != split for sample in samples):
        raise ValueError("Cache split role does not match its manifest rows")
    prefix = model.features[:7].eval()
    identity = {
        "format": 1,
        "manifest_sha256": file_sha256(manifest),
        "input_hashes": [file_sha256(sample.path) for sample in samples],
        "prefix_state_sha256": torch_state_sha256(prefix),
        "image_size": model.config.image_size,
        "model_type": model.config.model_type,
        "split": split,
        "views": views,
        "seed": seed,
        "dtype": "float32",
        "transform_source_sha256": file_sha256(image_data.__file__),
        "processor_source_sha256": file_sha256(image_processor.__file__),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pillow": PIL.__version__,
        "numpy": np.__version__,
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (split + "-" + key + ".npy")
    sidecar = path.with_suffix(".json")
    labels = tuple(sample.label for sample in samples)
    if path.exists() and sidecar.exists():
        saved = json.loads(sidecar.read_text(encoding="utf-8"))
        if saved["identity"] != identity or saved["sha256"] != file_sha256(path):
            raise ValueError("Feature cache checksum or identity mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != saved["shape"] or array.dtype != np.float32:
            raise ValueError("Invalid feature cache dimensions or dtype")
        return FeatureCache(path, key, labels, views)
    dataset = ImageManifestDataset(manifest, image_size=model.config.image_size, training=split == "train")
    temporary = path.with_suffix(".npy.part")
    with torch.inference_mode():
        shape = tuple(prefix(dataset[0]["pixel_values"][None]).shape[1:])
        array = np.lib.format.open_memmap(
            temporary, mode="w+", dtype=np.float32, shape=(views, len(dataset), *shape)
        )
        for view in range(views):
            seed_everything(seed + view)
            offset = 0
            loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
            for batch in loader:
                values = prefix(batch["pixel_values"]).cpu().numpy()
                if not np.isfinite(values).all():
                    raise ValueError("CNN prefix produced non-finite features")
                array[view, offset : offset + len(values)] = values
                offset += len(values)
                if offset % 128 == 0 or offset == len(dataset):
                    print(f"CACHE {split} view {view + 1}/{views}: {offset}/{len(dataset)}", flush=True)
        shape = list(array.shape)
        array.flush()
        del array
    if (
        file_sha256(manifest) != identity["manifest_sha256"]
        or [file_sha256(sample.path) for sample in samples] != identity["input_hashes"]
        or torch_state_sha256(prefix) != identity["prefix_state_sha256"]
        or file_sha256(image_data.__file__) != identity["transform_source_sha256"]
        or file_sha256(image_processor.__file__) != identity["processor_source_sha256"]
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError("Feature cache inputs or processor changed during extraction")
    temporary.replace(path)
    metadata = {"identity": identity, "key": key, "sha256": file_sha256(path), "shape": shape}
    pending = sidecar.with_suffix(".json.part")
    pending.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pending.replace(sidecar)
    return FeatureCache(path, key, labels, views)


def _loader(cache, *, view, batch_size, shuffle, seed):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    class CachedDataset(Dataset):
        def __init__(self):
            self.array = np.load(cache.path, mmap_mode="r", allow_pickle=False)

        def __len__(self):
            return len(cache.labels)

        def __getitem__(self, index):
            return torch.from_numpy(np.array(self.array[view, index], copy=True)), cache.labels[index]

    return DataLoader(
        CachedDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )


def _validation(model, cache, batch_size):
    import torch

    model.eval()
    labels, scores = [], []
    with torch.inference_mode():
        for features, targets in _loader(cache, view=0, batch_size=batch_size, shuffle=False, seed=0):
            logits = model.classifier(model.avgpool(model.features[7:](features)))
            scores.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
            labels.extend(targets.tolist())
    return labels, scores


def fit_cached(
    model,
    train_manifest,
    validation_manifest,
    output_dir,
    cache_dir,
    cfg: CachedTrainConfig,
    *,
    data_audit: dict,
) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score

    for manifest in (train_manifest, validation_manifest):
        if {sample.label for sample in load_image_samples(manifest)} != {0, 1}:
            raise ValueError("Training and tuning validation require both classes")
    prefix_hash = torch_state_sha256(model.features[:7])
    initial_hash = torch_state_sha256(model)
    train = ensure_feature_cache(model, train_manifest, cache_dir, split="train", views=2, seed=cfg.seed)
    validation = ensure_feature_cache(model, validation_manifest, cache_dir, split="val", seed=cfg.seed)
    seed_everything(cfg.seed)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": cfg.head_lr},
            {"params": model.features[7:].parameters(), "lr": cfg.backbone_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    steps = max(1, math.ceil(len(train.labels) / cfg.batch_size) * cfg.epochs)
    warmup = max(1, int(steps * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.0, (steps - step) / max(1, steps - warmup)))
    )
    weights = torch.tensor([cfg.real_loss_weight, 1.0])
    best_rank, best_state, best_epoch = None, None, -1
    history = []
    started = time.monotonic()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg.epochs):
        if epoch == cfg.freeze_epochs:
            for parameter in model.features[7:].parameters():
                parameter.requires_grad = True
        model.train()
        model.features[:7].eval()
        total_loss, count = 0.0, 0
        for features, targets in _loader(
            train, view=epoch % train.views, batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model.classifier(model.avgpool(model.features[7:](features)))
            loss = torch.nn.functional.cross_entropy(logits, targets, weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(targets)
            count += len(targets)
        labels, scores = _validation(model, validation, cfg.batch_size)
        metrics = compute_metrics(labels, scores)
        partial = float(roc_auc_score(labels, scores, max_fpr=0.05))
        rank = (partial, float(roc_auc_score(labels, scores)))
        history.append(
            {
                "epoch": epoch,
                "loss": total_loss / count,
                "val_partial_auc_005": partial,
                "val_metrics": metrics.to_dict(),
                "seconds": time.monotonic() - started,
            }
        )
        if best_rank is None or rank > best_rank:
            best_rank, best_epoch = rank, epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        pending = out / "history.json.part"
        pending.write_text(json.dumps(history, indent=2), encoding="utf-8")
        pending.replace(out / "history.json")
        print(
            f"EPOCH {epoch + 1}/{cfg.epochs} loss={total_loss / count:.4f} "
            f"tuning pAUC@5%={partial:.4f} AUC={metrics.auc} "
            f"accuracy={metrics.accuracy} FPR={metrics.false_positive_rate}",
            flush=True,
        )
    if best_state is None or best_rank is None:
        raise RuntimeError("Training produced no selected checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    if torch_state_sha256(model.features[:7]) != prefix_hash:
        raise ValueError("Frozen prefix changed; cached features no longer represent the model")
    labels, scores = _validation(model, validation, cfg.batch_size)
    selected_metrics = compute_metrics(labels, scores)
    model.save_pretrained(out)
    (out / "data-audit.json").write_text(json.dumps(data_audit, indent=2), encoding="utf-8")
    write_metrics(
        selected_metrics,
        out / "metrics.json",
        extra={
            "evaluation_split": "val",
            "test_evaluated": False,
            "architecture": "efficientnet_b0",
            "optimizer": "AdamW",
            "loss": "class-weighted cross_entropy",
            "config": asdict(cfg),
            "selected_epoch": best_epoch,
            "val_partial_auc_005": best_rank[0],
            "data_kind": data_audit["data_kind"],
            "dataset_fingerprint": data_audit["fingerprint"],
            "torch_state_sha256": torch_state_sha256(model),
            "image_size": model.config.image_size,
            "initial_state_sha256": initial_hash,
            "prefix_state_sha256": prefix_hash,
            "train_cache_key": train.key,
            "val_cache_key": validation.key,
            "train_size": len(train.labels),
            "val_size": len(validation.labels),
            "torch_version": torch.__version__,
            "history": history,
            "wall_time_sec": time.monotonic() - started,
        },
    )
    return {
        "output_dir": str(out),
        "best_partial_auc": best_rank[0],
        "best_auc": best_rank[1],
        "best_epoch": best_epoch,
        "history": history,
    }
