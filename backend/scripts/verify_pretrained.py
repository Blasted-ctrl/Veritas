"""Compare serving preprocessing and ONNX predictions against an offline Torch reference."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from transformers import ViTForImageClassification, ViTImageProcessor

    from veritas.api.inference import ImageDetector
    from veritas.models.metadata import file_sha256

    torch.set_num_threads(2)
    detector = ImageDetector(args.model)
    if not detector.available:
        raise ValueError("ONNX detector is not available")
    processor = ViTImageProcessor.from_pretrained(args.reference_dir, local_files_only=True)
    reference = ViTForImageClassification.from_pretrained(args.reference_dir, local_files_only=True).eval()
    if reference.config.id2label != {0: "REAL", 1: "FAKE"}:
        raise ValueError("Reference labels do not match the pinned detector")
    rows = []
    for image_path in args.images:
        data = Path(image_path).read_bytes()
        with Image.open(io.BytesIO(data)) as image:
            expected = processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"]
        served = detector.preprocess(data)
        np.testing.assert_allclose(served, expected.numpy(), atol=1e-6, rtol=0)
        with torch.inference_mode():
            logits = reference(pixel_values=expected).logits.numpy()
        actual_logits = detector.session.run(["logits"], {"pixel_values": served})[0]
        np.testing.assert_allclose(actual_logits, logits, atol=1e-4, rtol=1e-4)
        expected_score = torch.softmax(torch.from_numpy(logits), dim=-1)[0, 1].item()
        actual_score = detector.fake_probability(data)
        np.testing.assert_allclose(actual_score, expected_score, atol=1e-5, rtol=0)
        rows.append(
            {
                "file": Path(image_path).name,
                "preprocessing_max_abs_error": float(np.max(np.abs(served - expected.numpy()))),
                "logits_max_abs_error": float(np.max(np.abs(actual_logits - logits))),
                "torch_fake_score": expected_score,
                "onnx_fake_score": actual_score,
            }
        )
        print(Path(image_path).name, actual_score, flush=True)
    report = {
        "onnx_sha256": file_sha256(args.model),
        "reference_sha256": file_sha256(Path(args.reference_dir) / "model.safetensors"),
        "reference_revision": "a6661e07d38f1a097bba07ca9415538819278f09",
        "processor_atol": 1e-6,
        "logits_atol": 1e-4,
        "score_atol": 1e-5,
        "torch_version": torch.__version__,
        "images": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Parity checks passed", flush=True)


if __name__ == "__main__":
    main()
