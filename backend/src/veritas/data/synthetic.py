"""Generate tiny but *real* media fixtures with stdlib only.

Used to (a) exercise the full prepare-data pipeline without downloading any
gated dataset, and (b) give CI deterministic, leak-aware fixtures.  Crucially
each synthetic subject appears as BOTH a real and a fake sample, which is the
exact condition under which identity leakage must be prevented.

Images are written as valid PNGs via a minimal ``zlib``/``struct`` encoder;
audio as valid WAVs via the stdlib :mod:`wave` module.  No Pillow / numpy /
soundfile required.
"""

from __future__ import annotations

import math
import random
import struct
import wave
import zlib
from pathlib import Path

from veritas.data.manifest import FAKE, REAL, Sample


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    """Write an RGB PNG from a 2-D list of (r, g, b) tuples."""
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    raw = bytearray()
    for row in pixels:
        raw.append(0)  # filter type 0 (None) for this scanline
        for r, g, b in row:
            raw += bytes((r & 0xFF, g & 0xFF, b & 0xFF))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def write_wav(path: Path, samples: list[int], sample_rate: int = 16000) -> None:
    """Write a 16-bit mono WAV from integer PCM samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(struct.pack("<h", max(-32768, min(32767, s))) for s in samples))


def _make_image(rng: random.Random, fake: bool, size: int = 32) -> list[list[tuple[int, int, int]]]:
    # REAL: smooth gradient. FAKE: gradient + high-frequency noise (a crude
    # stand-in for manipulation artefacts).  Pixel content is irrelevant to the
    # Phase-1 splitter; this just produces distinguishable, valid files.
    base = rng.randint(40, 200)
    pixels = []
    for y in range(size):
        row = []
        for x in range(size):
            v = (base + x * 3 + y * 2) % 256
            if fake:
                v = (v + rng.randint(-60, 60)) % 256
            row.append((v, (v * 2) % 256, (v * 3) % 256))
        pixels.append(row)
    return pixels


def _make_audio(rng: random.Random, fake: bool, sample_rate: int = 16000, seconds: float = 0.25) -> list[int]:
    n = int(sample_rate * seconds)
    freq = rng.uniform(110, 330)
    out = []
    for i in range(n):
        v = math.sin(2 * math.pi * freq * i / sample_rate)
        if fake:
            # Inject a metallic harmonic + jitter, a toy "synthetic voice" cue.
            v = 0.6 * v + 0.4 * math.sin(2 * math.pi * freq * 4 * i / sample_rate)
            v += rng.uniform(-0.05, 0.05)
        out.append(int(v * 12000))
    return out


def generate_dataset(
    output_dir: Path,
    modality: str,
    *,
    subjects: int = 24,
    per_subject: int = 4,
    seed: int = 1337,
) -> list[Sample]:
    """Create a balanced synthetic dataset and return its manifest.

    Each subject contributes ``per_subject`` samples, split evenly between REAL
    and FAKE, so the dataset is class-balanced and every identity spans both
    labels (the leakage-prone arrangement we must split safely).
    """
    if modality not in ("image", "audio"):
        raise ValueError(f"modality must be 'image' or 'audio', got {modality!r}")
    if per_subject % 2 != 0:
        raise ValueError("per_subject must be even so each subject is class-balanced")

    rng = random.Random(seed)
    raw_dir = Path(output_dir)
    ext = "png" if modality == "image" else "wav"
    samples: list[Sample] = []

    for s in range(subjects):
        subject_id = f"subject_{s:04d}"
        for k in range(per_subject):
            fake = k >= per_subject // 2
            label = FAKE if fake else REAL
            rel = Path(label_dirname(label)) / f"{subject_id}_{k:02d}.{ext}"
            dest = raw_dir / rel
            if modality == "image":
                write_png(dest, _make_image(rng, fake))
            else:
                write_wav(dest, _make_audio(rng, fake))
            samples.append(
                Sample(
                    path=str(dest),
                    label=label,
                    subject_id=subject_id,
                    modality=modality,
                    source="synthetic",
                )
            )
    return samples


def label_dirname(label: int) -> str:
    return "real" if label == REAL else "fake"


__all__ = ["generate_dataset", "write_png", "write_wav", "label_dirname"]
