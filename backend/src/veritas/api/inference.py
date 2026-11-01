"""ONNX Runtime inference for the image and audio detectors.

Preprocessing is reimplemented in NumPy (mirroring the torch training
transforms) so the serving path depends only on onnxruntime + numpy + Pillow +
soundfile — not torch. Each detector loads its ONNX graph lazily and reports
whether it is available so the API can degrade gracefully.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from threading import Lock

import numpy as np

from veritas.api.cache import content_hash
from veritas.models.metadata import default_image_metadata, file_sha256, load_image_metadata

logger = logging.getLogger("veritas.inference")


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class _OnnxModel:
    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._session = None
        self._load_lock = Lock()
        self._failed = False
        self.metadata = {"id": "Custom audio model", "source_type": "custom"}

    @property
    def available(self) -> bool:
        if not self.model_path.is_file() or self._failed:
            return False
        try:
            return self.session is not None
        except Exception as exc:
            self._failed = True
            logger.warning("Model could not be loaded: %s", exc)
            return False

    def _prepare(self) -> None:
        self.metadata["artifact_sha256"] = file_sha256(self.model_path)

    def _validate_session(self, session) -> None:
        outputs = session.get_outputs()
        if len(outputs) != 1 or outputs[0].name != "logits" or outputs[0].shape[-1] != 2:
            raise ValueError("Expected a binary classifier with a logits output")

    @property
    def session(self):
        if self._session is None:
            with self._load_lock:
                if self._session is None:
                    import onnxruntime as ort

                    self._prepare()
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 2
                    session = ort.InferenceSession(
                        str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"]
                    )
                    self._validate_session(session)
                    self._session = session
        return self._session

    @property
    def fingerprint(self) -> str:
        if not self.available:
            raise ValueError("The model is unavailable")
        return content_hash(json.dumps(self.metadata, sort_keys=True).encode("utf-8"))

    @property
    def model_info(self) -> dict:
        return {
            key: self.metadata[key]
            for key in ("id", "source_type", "source_url", "revision", "license", "artifact_sha256")
            if key in self.metadata
        }

    def _run(self, input_name: str, array: np.ndarray) -> np.ndarray:
        logits = self.session.run(["logits"], {input_name: array})[0]
        if logits.shape != (array.shape[0], 2) or not np.isfinite(logits).all():
            raise ValueError("The model returned invalid classification scores")
        return _softmax(logits)


def preprocess_image(image_bytes: bytes, metadata: dict) -> np.ndarray:
    from PIL import Image

    size = metadata["image_size"]
    with Image.open(io.BytesIO(image_bytes)) as img:
        image = img.convert("RGB").resize((size, size), Image.Resampling(metadata["resample"]))
        arr = np.asarray(image, dtype=np.float32) * np.float32(metadata["rescale_factor"])
    arr = (arr - np.array(metadata["mean"], dtype=np.float32)) / np.array(metadata["std"], dtype=np.float32)
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


class ImageDetector(_OnnxModel):
    def __init__(self, model_path: str | Path, *, image_size: int = 224):
        super().__init__(model_path)
        self.image_size = image_size
        self.metadata = default_image_metadata(image_size)

    def _prepare(self) -> None:
        self.metadata = load_image_metadata(self.model_path, image_size=self.image_size)
        self.image_size = self.metadata["image_size"]
        super()._prepare()

    def _validate_session(self, session) -> None:
        super()._validate_session(session)
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "pixel_values":
            raise ValueError("Expected pixel_values image input")
        shape = inputs[0].shape
        if len(shape) != 4 or any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(shape[1:], [3, self.image_size, self.image_size], strict=True)
        ):
            raise ValueError("Model image dimensions do not match the processor metadata")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        if self._session is None:
            self.metadata = load_image_metadata(self.model_path, image_size=self.image_size)
            self.image_size = self.metadata["image_size"]
        return preprocess_image(image_bytes, self.metadata)

    def fake_probability(self, image_bytes: bytes) -> float:
        probs = self._run("pixel_values", self.preprocess(image_bytes))
        return float(probs[0, self.metadata["fake_index"]])

    def fake_probability_array(self, arr: np.ndarray) -> float:
        return float(self._run("pixel_values", arr)[0, self.metadata["fake_index"]])


class AudioDetector(_OnnxModel):
    def __init__(self, model_path: str | Path, *, sample_rate: int = 16000):
        super().__init__(model_path)
        self.sample_rate = sample_rate

    def _prepare(self) -> None:
        self.metadata["sample_rate"] = self.sample_rate
        super()._prepare()

    def _validate_session(self, session) -> None:
        super()._validate_session(session)
        inputs = session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != "input_values" or len(inputs[0].shape) != 2:
            raise ValueError("Expected a mono input_values audio input")

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
    if not np.isfinite(fake_prob) or not 0 <= fake_prob <= 1:
        raise ValueError("Invalid model probability")
    verdict = "fake" if fake_prob >= threshold else "real"
    confidence = fake_prob if verdict == "fake" else 1.0 - fake_prob
    return verdict, float(confidence)


__all__ = ["ImageDetector", "AudioDetector", "verdict_from_probability"]
