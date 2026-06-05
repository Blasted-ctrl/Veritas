"""PyTorch dataset that materialises a Phase 1 audio manifest into waveforms."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from veritas.data.manifest import read_manifest
from veritas.models.audio_model import TARGET_SAMPLE_RATE, load_audio


class AudioManifestDataset(Dataset):
    """Loads audio listed in a manifest, resamples to 16 kHz and normalizes.

    Each item is ``{"input_values": FloatTensor[T], "labels": int}``. Waveforms
    are zero-mean/unit-variance normalized per utterance (Wav2Vec2 convention)
    and optionally truncated to ``max_seconds`` to bound memory.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        target_sr: int = TARGET_SAMPLE_RATE,
        max_seconds: float = 4.0,
        limit: int | None = None,
    ):
        self.samples = read_manifest(manifest_path)
        if limit is not None:
            self.samples = self.samples[:limit]
        self.target_sr = target_sr
        self.max_len = int(max_seconds * target_sr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        waveform = load_audio(sample.path, self.target_sr)
        if waveform.numel() > self.max_len:
            waveform = waveform[: self.max_len]
        # Per-utterance normalization (zero mean, unit variance).
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
        return {"input_values": waveform, "labels": sample.label}


def collate(batch: list[dict]) -> dict:
    """Right-pad variable-length waveforms and build an attention mask."""
    import torch

    lengths = [b["input_values"].shape[0] for b in batch]
    max_len = max(lengths)
    input_values = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_values"].shape[0]
        input_values[i, :n] = b["input_values"]
        attention_mask[i, :n] = 1
    labels = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
    return {"input_values": input_values, "attention_mask": attention_mask, "labels": labels}


__all__ = ["AudioManifestDataset", "collate"]
