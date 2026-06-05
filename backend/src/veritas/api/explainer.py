"""Optional Grad-CAM explainer for the API.

Grad-CAM needs gradients, so this loads the *torch* ViT (from a ``save_pretrained``
directory) lazily. If torch or the model directory is unavailable the explainer
reports ``available == False`` and the API simply omits the heatmap — the ONNX
verdict path is unaffected.
"""

from __future__ import annotations

from pathlib import Path


class GradCamExplainer:
    def __init__(self, model_dir: str, *, image_size: int = 224):
        self.model_dir = Path(model_dir)
        self.image_size = image_size
        self._model = None

    @property
    def available(self) -> bool:
        if not (self.model_dir / "config.json").exists():
            return False
        try:
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

    def explain(self, image_bytes: bytes, *, class_idx: int = 1) -> dict | None:
        if not self.available:
            return None
        from veritas.explain.gradcam import explain_image

        return explain_image(self._load(), image_bytes, image_size=self.image_size, class_idx=class_idx)


__all__ = ["GradCamExplainer"]
