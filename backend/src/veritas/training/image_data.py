"""Manifest-backed image tensors with separate training and evaluation transforms."""

from __future__ import annotations

import io
import random
from pathlib import Path

from torch.utils.data import Dataset

from veritas.data.image_corpus import load_image_samples
from veritas.models.image_model import IMAGENET_MEAN, IMAGENET_STD, make_image_transform


class JPEGRecompression:
    def __call__(self, image):
        from PIL import Image

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(65, 100))
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


def make_training_transform(image_size: int):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([JPEGRecompression()], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ImageManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        image_size: int = 224,
        limit: int | None = None,
        training: bool = False,
        seed: int = 1337,
    ):
        self.samples = load_image_samples(manifest_path)
        if limit is not None:
            if limit < 2:
                raise ValueError("A limited training/evaluation split needs at least two samples")
            if limit < len(self.samples):
                groups = {label: [s for s in self.samples if s.label == label] for label in (0, 1)}
                if not all(groups.values()):
                    raise ValueError("Limited splits must contain both classes")
                rng = random.Random(seed)
                for group in groups.values():
                    rng.shuffle(group)
                fake = max(
                    1, min(round(limit * len(groups[1]) / len(self.samples)), limit - 1, len(groups[1]))
                )
                real = min(limit - fake, len(groups[0]))
                self.samples = groups[0][:real] + groups[1][: limit - real]
                rng.shuffle(self.samples)
        self.transform = make_training_transform(image_size) if training else make_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from PIL import Image

        sample = self.samples[idx]
        with Image.open(sample.path) as img:
            pixel_values = self.transform(img.convert("RGB"))
        return {"pixel_values": pixel_values, "labels": sample.label}


def collate(batch: list[dict]) -> dict:
    import torch

    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
    }
