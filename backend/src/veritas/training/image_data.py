"""PyTorch dataset that materialises a Phase 1 image manifest into tensors."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from veritas.data.manifest import read_manifest
from veritas.models.image_model import make_image_transform


class ImageManifestDataset(Dataset):
    """Loads images listed in a manifest CSV and applies the ViT transform.

    Each item is ``{"pixel_values": FloatTensor[3, H, W], "labels": int}``.
    """

    def __init__(self, manifest_path: str | Path, *, image_size: int = 224, limit: int | None = None):
        from PIL import Image  # noqa: F401  (validate availability early)

        self.samples = read_manifest(manifest_path)
        if limit is not None:
            self.samples = self.samples[:limit]
        self.transform = make_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            image = img.convert("RGB")
            pixel_values = self.transform(image)
        return {"pixel_values": pixel_values, "labels": sample.label}


def collate(batch: list[dict]) -> dict:
    """Stack a list of items into a batch dict for the model."""
    import torch

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }


__all__ = ["ImageManifestDataset", "collate"]
