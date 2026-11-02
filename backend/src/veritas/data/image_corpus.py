"""Preflight checks for curated image manifests and generator holdouts."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import replace
from pathlib import Path

from veritas.data.manifest import SPLITS, Sample, label_counts, read_manifest
from veritas.models.metadata import file_sha256


def load_image_samples(manifest: str | Path) -> list[Sample]:
    manifest = Path(manifest)
    samples = []
    for sample in read_manifest(manifest):
        path = Path(sample.path).expanduser()
        if not path.is_absolute():
            relative = manifest.parent / path
            path = relative if relative.exists() else path
        samples.append(
            replace(
                sample,
                path=str(path.resolve(strict=True)),
                subject_id=sample.subject_id.strip(),
                generator_id=sample.generator_id.strip(),
            )
        )
    return samples


def image_identity(path: str | Path) -> tuple[str, str]:
    """Encoded and decoded RGB hashes; metadata-only re-encoding cannot hide duplicates."""
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        pixels = hashlib.sha256(struct.pack("!II", *rgb.size) + rgb.tobytes()).hexdigest()
    return file_sha256(path), pixels


def audit_image_corpus(directory: str | Path, *, heldout_generators=()) -> dict:
    directory = Path(directory)
    holdouts = sorted(set(heldout_generators))
    groups: dict[str, str] = {}
    encoded: dict[str, tuple[str, int]] = {}
    decoded: dict[str, tuple[str, int]] = {}
    observed_generators: set[str] = set()
    sources: set[str] = set()
    digest = hashlib.sha256()
    report: dict = {"splits": {}, "heldout_generators": holdouts}
    for split in SPLITS:
        manifest = directory / (split + ".csv")
        digest.update(file_sha256(manifest).encode())
        samples = load_image_samples(manifest)
        counts = label_counts(samples)
        if not all(counts.values()):
            raise ValueError(f"{split} must contain both real and fake images")
        generators: dict[str, int] = {}
        for sample in samples:
            if sample.modality != "image" or sample.split not in ("", split):
                raise ValueError(f"Invalid modality or split in {manifest.name}")
            group = sample.subject_id.strip()
            if not group or groups.setdefault(group, split) != split:
                raise ValueError(f"Image group crosses split boundaries: {group}")
            if sample.label == 1:
                generator = sample.generator_id.strip()
                if holdouts and not generator:
                    raise ValueError(
                        "Generator provenance is required for all fakes when using held-out generators"
                    )
                observed_generators.add(generator)
                generators[generator or "unknown"] = generators.get(generator or "unknown", 0) + 1
                if generator in holdouts and split != "test":
                    raise ValueError(f"The held-out generator {generator} occurs in {split}")
            raw_hash, pixels = image_identity(sample.path)
            for key, seen, kind in (
                (raw_hash, encoded, "encoded image"),
                (pixels, decoded, "decoded pixels"),
            ):
                previous = seen.setdefault(key, (split, sample.label))
                if previous[1] != sample.label:
                    raise ValueError(f"Conflicting labels for identical {kind}")
                if previous[0] != split:
                    raise ValueError(f"Identical {kind} cross split boundaries")
            digest.update(pixels.encode())
            sources.add(sample.source)
        report["splits"][split] = {
            "samples": len(samples),
            "real": counts[0],
            "fake": counts[1],
            "groups": len({sample.subject_id for sample in samples}),
            "generators": generators,
            "sources": sorted({sample.source for sample in samples}),
        }
    absent = set(holdouts) - observed_generators
    if absent:
        raise ValueError(f"Requested held-out generators are absent: {sorted(absent)}")
    report["fingerprint"] = digest.hexdigest()
    report["data_kind"] = (
        "procedural-fixture"
        if sources == {"synthetic"}
        else "mixed-with-fixtures"
        if "synthetic" in sources
        else "curated-images"
    )
    report["checks"] = ["group separation", "encoded content separation", "decoded pixel separation"]
    report["limits"] = (
        "Exact duplicates only; curator group and generator identities are not independently verified."
    )
    return report
