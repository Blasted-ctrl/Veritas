"""Evaluate fixed held-out examples through the same ONNX processor as the API."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path

from veritas.data.image_corpus import audit_image_corpus, image_identity, load_image_samples
from veritas.data.manifest import read_manifest
from veritas.models.metadata import file_sha256
from veritas.training.metrics import compute_metrics

VARIANTS = ("original", "jpeg75", "downscale50")


def _variant_bytes(raw: bytes, variant: str) -> bytes:
    from PIL import Image

    if variant == "original":
        return raw
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        if variant == "jpeg75":
            image.save(buffer, format="JPEG", quality=75, subsampling=2)
        elif variant == "downscale50":
            size = tuple(max(1, dimension // 2) for dimension in image.size)
            image.resize(size, Image.Resampling.BILINEAR).save(buffer, format="PNG")
        else:
            raise ValueError(f"Unknown evaluation variant: {variant}")
        return buffer.getvalue()


def _group_metrics(predictions, field):
    groups = {}
    for row in predictions:
        if field == "generator_id" and row["label"] == 0:
            continue
        groups.setdefault(row[field] or "unknown", []).append(row)
    return {
        name: compute_metrics(
            [row["label"] for row in rows],
            [row["fake_score"] for row in rows],
            threshold=rows[0]["threshold"],
        ).to_dict()
        for name, rows in sorted(groups.items())
    }


def evaluate_image(
    model_path: str | Path,
    manifest: str | Path,
    *,
    output: str | Path | None = None,
    training_dir: str | Path | None = None,
    threshold: float = 0.5,
) -> dict:
    """Report measured scores without tuning thresholds or selecting checkpoints.

    Optional training_dir checks train/validation groups and exact content. It
    cannot establish separation from undisclosed upstream training datasets.
    """
    from veritas.api.inference import ImageDetector

    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    manifest = Path(manifest)
    samples = load_image_samples(manifest)
    if not samples or any(row.split != "test" or row.modality != "image" for row in samples):
        raise ValueError("Evaluation requires nonempty image rows explicitly marked test")
    forbidden_groups = set()
    forbidden_content = set()
    reference = None
    if training_dir is not None:
        reference = audit_image_corpus(training_dir)
        for split in ("train", "val"):
            for row in load_image_samples(Path(training_dir) / (split + ".csv")):
                forbidden_groups.add(row.subject_id.strip())
                forbidden_content.update(image_identity(row.path))
    identities = []
    observed = set()
    for row in samples:
        if not row.subject_id.strip() or row.subject_id.strip() in forbidden_groups:
            raise ValueError("Evaluation group overlaps training/validation or is empty")
        identity = image_identity(row.path)
        if forbidden_content.intersection(identity):
            raise ValueError("Evaluation content overlaps training/validation")
        if observed.intersection(identity):
            raise ValueError("Duplicate evaluation content would inflate the sample count")
        observed.update(identity)
        identities.append(identity)
    detector = ImageDetector(model_path)
    if not detector.available:
        raise ValueError("Image model is unavailable or its processing metadata is invalid")
    trained_on = detector.metadata.get("training_data_fingerprint")
    if reference is not None and trained_on and trained_on != reference["fingerprint"]:
        raise ValueError("Reference corpus does not match model training provenance")
    digest = hashlib.sha256(file_sha256(manifest).encode())
    for raw_hash, pixels in identities:
        digest.update(raw_hash.encode())
        digest.update(pixels.encode())
    report = {
        "schema_version": 1,
        "model": detector.model_info,
        "processor": detector.metadata,
        "manifest_sha256": file_sha256(manifest),
        "evaluation_fingerprint": digest.hexdigest(),
        "threshold": threshold,
        "separation": {
            "training_reference_checked": reference is not None,
            "reference_fingerprint": reference["fingerprint"] if reference else None,
            "model_provenance_matched": bool(reference and trained_on),
        },
        "limits": [
            "Curator labels, group IDs and generator identities are not independently verified.",
            "Exact-content checks do not detect near duplicates or undisclosed upstream training overlap.",
            "Diagnostic examples and small samples do not estimate general detection accuracy.",
            "Generator groups contain fakes only; their accuracy is recall, not balanced accuracy.",
            "Each variant is the same examples transformed; variants are not independent observations.",
        ],
        "variants": {},
    }
    original_rows = read_manifest(manifest)
    for variant in VARIANTS:
        predictions = []
        for row, original in zip(samples, original_rows, strict=True):
            score = detector.fake_probability(_variant_bytes(Path(row.path).read_bytes(), variant))
            predictions.append(
                {
                    "path": original.path,
                    "label": row.label,
                    "subject_id": row.subject_id,
                    "source": row.source,
                    "generator_id": row.generator_id,
                    "source_url": row.source_url,
                    "fake_score": score,
                    "prediction": int(score >= threshold),
                    "threshold": threshold,
                }
            )
        report["variants"][variant] = {
            "metrics": compute_metrics(
                [row.label for row in samples],
                [row["fake_score"] for row in predictions],
                threshold=threshold,
            ).to_dict(),
            "by_source": _group_metrics(predictions, "source"),
            "by_generator": _group_metrics(predictions, "generator_id"),
            "predictions": predictions,
        }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
