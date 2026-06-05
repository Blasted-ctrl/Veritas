"""ONNX Runtime inference for the image and audio detectors.

Preprocessing is reimplemented in NumPy (mirroring the torch training
transforms) so the serving path depends only on onnxruntime + numpy + Pillow +
soundfile — not torch. Each detector loads its ONNX graph lazily and reports
whether it is available so the API can degrade gracefully.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class _OnnxModel:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._session = None

    @property
    def available(self) -> bool:
        return self.model_path.exists()

    @property
    def session(self):
        if self._session is None:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        return self._session

    def _run(self, input_name: str, array: np.ndarray) -> np.ndarray:
        logits = self.session.run(["logits"], {input_name: array})[0]
        return _softmax(logits)


class ImageDetector(_OnnxModel):
    def __init__(self, model_path: str | Path, *, image_size: int = 224):
        super().__init__(model_path)
        self.image_size = image_size

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            image = img.convert("RGB").resize((self.image_size, self.image_size))
            arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # [1, 3, H, W]
        return arr.astype(np.float32)

    def fake_probability(self, image_bytes: bytes) -> float:
        probs = self._run("pixel_values", self.preprocess(image_bytes))
        return float(probs[0, 1])

    def fake_probability_array(self, arr: np.ndarray) -> float:
        return float(self._run("pixel_values", arr)[0, 1])


class AudioDetector(_OnnxModel):
    def __init__(self, model_path: str | Path, *, sample_rate: int = 16000):
        super().__init__(model_path)
        self.sample_rate = sample_rate

    def preprocess(self, audio_bytes: bytes) -> np.ndarray:
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        waveform = data.mean(axis=1)  # mono
        if sr != self.sample_rate and waveform.size > 1:
            # Lightweight linear resample (avoids a torch/soxr dependency at
            # serve time; inputs are typically already 16 kHz).
            duration = waveform.size / sr
            target_n = max(1, int(round(duration * self.sample_rate)))
            xp = np.linspace(0.0, 1.0, num=waveform.size, endpoint=False)
            x = np.linspace(0.0, 1.0, num=target_n, endpoint=False)
            waveform = np.interp(x, xp, waveform).astype(np.float32)
        # Per-utterance normalization.
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
        return waveform[np.newaxis, :].astype(np.float32)  # [1, T]

    def fake_probability(self, audio_bytes: bytes) -> float:
        probs = self._run("input_values", self.preprocess(audio_bytes))
        return float(probs[0, 1])


def verdict_from_probability(fake_prob: float, threshold: float = 0.5) -> tuple[str, float]:
    """Map a fake-probability to a ``(verdict, confidence)`` pair."""
    verdict = "fake" if fake_prob >= threshold else "real"
    confidence = fake_prob if verdict == "fake" else 1.0 - fake_prob
    return verdict, float(confidence)


__all__ = ["ImageDetector", "AudioDetector", "verdict_from_probability"]
