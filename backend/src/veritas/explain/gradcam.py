"""Grad-CAM interpretability for the ViT image detector.

Produces a heatmap over the input image highlighting the regions that most
drove the model's verdict. Grad-CAM needs gradients, so this uses the *torch*
model (not the ONNX graph): we hook the final transformer block, weight its
token activations by their gradients w.r.t. the target-class score, reshape the
patch tokens back to a 2-D grid, and upsample to the image.

Requires the optional ``ml`` extra (torch + Pillow).
"""

from __future__ import annotations

import base64
import io

import numpy as np

from veritas.models.image_model import make_image_transform


def _last_encoder_block(model):
    """Return the final ViT encoder block across transformers naming schemes."""
    vit = model.vit
    if hasattr(vit, "layers"):  # transformers >= 5.x
        return vit.layers[-1]
    return vit.encoder.layer[-1]  # transformers < 5.x


class ViTGradCAM:
    """Grad-CAM for ``ViTForImageClassification``."""

    def __init__(self, model, target_layer=None):
        self.model = model.eval()
        self.target = target_layer or _last_encoder_block(model)
        self._activations = None
        self._gradients = None
        self.target.register_forward_hook(self._save_activation)
        self.target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inp, output):
        self._activations = output[0] if isinstance(output, tuple) else output

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0]

    def heatmap(self, pixel_values, class_idx: int = 1) -> np.ndarray:
        """Return a ``[H, W]`` heatmap in ``[0, 1]`` for ``class_idx``."""
        import torch

        self.model.zero_grad()
        logits = self.model(pixel_values=pixel_values).logits
        score = logits[:, class_idx].sum()
        score.backward()

        acts = self._activations  # [B, T, C]
        grads = self._gradients  # [B, T, C]
        # Grad-CAM for transformers: channel-summed grad*activation per token.
        cam = (grads * acts).sum(dim=-1)[0]  # [T]
        cam = cam[1:]  # drop the CLS token -> one weight per patch

        n_patches = cam.shape[0]
        side = int(round(n_patches**0.5))
        grid = cam[: side * side].reshape(side, side)
        grid = torch.relu(grid)
        if grid.max() > 0:
            grid = grid / grid.max()

        size = pixel_values.shape[-1]
        heat = _resize_2d(grid.detach().cpu().numpy().astype(np.float32), size, size)
        return heat


def _resize_2d(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    img = img.resize((width, height), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _jet(values: np.ndarray) -> np.ndarray:
    """Map ``[H, W]`` in [0,1] to an RGB uint8 jet-style colormap."""
    v = np.clip(values, 0, 1)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_png(image_bytes: bytes, heatmap: np.ndarray, *, alpha: float = 0.5) -> bytes:
    """Blend ``heatmap`` over the original image; return PNG bytes."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        base = im.convert("RGB").resize((heatmap.shape[1], heatmap.shape[0]))
    base_arr = np.asarray(base, dtype=np.float32)
    color = _jet(heatmap).astype(np.float32)
    blended = (1 - alpha) * base_arr + alpha * color
    out = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def explain_image(model, image_bytes: bytes, *, image_size: int = 224, class_idx: int = 1) -> dict:
    """Full Grad-CAM for one image: returns base64 heatmap + overlay data URLs."""
    import torch
    from PIL import Image

    transform = make_image_transform(image_size)
    with Image.open(io.BytesIO(image_bytes)) as im:
        pixel_values = transform(im.convert("RGB")).unsqueeze(0)

    cam = ViTGradCAM(model)
    with torch.enable_grad():
        heat = cam.heatmap(pixel_values, class_idx=class_idx)

    heat_png = overlay_png(image_bytes, heat, alpha=1.0)  # pure heatmap
    overlay = overlay_png(image_bytes, heat, alpha=0.5)  # blended overlay
    return {
        "heatmap": _png_data_url(heat_png),
        "overlay": _png_data_url(overlay),
        "grid_size": int(round((cam._activations.shape[1] - 1) ** 0.5)),
    }


__all__ = ["ViTGradCAM", "explain_image", "overlay_png"]
