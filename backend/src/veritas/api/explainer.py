"""Optional Grad-CAM explainer for the API.

Grad-CAM needs gradients, so this loads the *torch* ViT (from a ``save_pretrained``
directory) lazily. If torch or the model directory is unavailable the explainer
reports ``available == False`` and the API simply omits the heatmap — the ONNX
verdict path is unaffected.
"""

from __future__ import annotations

import json
from pathlib import Path

from veritas.models.metadata import torch_state_sha256


class GradCamExplainer:
    def __init__(self, model_dir: str, *, image_size: int = 224):
        self.model_dir = Path(model_dir)
        self.image_size = image_size
        self._model = None
        self._state_hash = None

    @property
    def available(self) -> bool:
        if not (self.model_dir / "config.json").exists():
            return False
        try:
            config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
            if config.get("model_type") != "vit":
                return False
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception:
            return False
        return True

    def _load(self):
        if self._model is None:
            from transformers import ViTForImageClassification

            self._model = ViTForImageClassification.from_pretrained(str(self.model_dir)).eval()
        return self._model

    def matches(self, preprocessing: dict) -> bool:
        expected = preprocessing.get("torch_state_sha256")
        if not expected or not self.available:
            return False
        try:
            if self._state_hash is None:
                self._state_hash = torch_state_sha256(self._load())
            return self._state_hash == expected
        except Exception:
            return False

    def explain(
        self, image_bytes: bytes, *, class_idx: int = 1, preprocessing: dict | None = None
    ) -> dict | None:
        if not self.available:
            return None
        from veritas.explain.gradcam import explain_image

        model = self._load()
        if preprocessing is not None:
            expected = preprocessing.get("torch_state_sha256")
            if not expected:
                return None
            if self._state_hash is None:
                self._state_hash = torch_state_sha256(model)
            if self._state_hash != expected:
                return None
        return explain_image(
            model, image_bytes, image_size=self.image_size, class_idx=class_idx, preprocessing=preprocessing
        )


__all__ = ["GradCamExplainer"]
