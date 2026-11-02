"""Deterministic source-balanced sampling with prompt and source-image exclusion."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter


def prompt_group(row: dict) -> str:
    prompt = " ".join(unicodedata.normalize("NFKC", row.get("prompt") or "").casefold().split())
    identity = "prompt:" + prompt if prompt else "image:" + row["image_sha256"]
    return hashlib.sha256(identity.encode()).hexdigest()


def select_rows(
    records: list[dict],
    *,
    real_quotas: dict[str, int],
    fake_count: int,
    fake_model_cap: int,
    seed: int = 1337,
    excluded_groups=(),
    excluded_hashes=(),
    required_models: dict[str, int] | None = None,
) -> list[dict]:
    if not real_quotas or min(real_quotas.values()) < 1 or fake_count < 1 or fake_model_cap < 1:
        raise ValueError("Source and generator quotas must be positive")
    required_models = required_models or {}
    if any(count < 1 or count > fake_model_cap for count in required_models.values()):
        raise ValueError("Required generator quota exceeds its cap")
    if sum(required_models.values()) > fake_count:
        raise ValueError("Required generator quotas exceed fake count")
    ordered = sorted(records, key=lambda row: hashlib.sha256(f"{seed}:{row['row_id']}".encode()).hexdigest())
    chosen = []
    real_counts: Counter = Counter()
    model_counts: Counter = Counter()
    blocked = set(excluded_groups)
    seen_hashes = set(excluded_hashes)
    seen_ids = set()
    for required in (*required_models, None):
        for row in ordered:
            label, model = row.get("label"), row.get("model", "").strip().casefold()
            digest = row.get("image_sha256", "")
            if label not in ("real", "fake") or not model or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("Invalid source label, model, or original image hash")
            if row["row_id"] in seen_ids or digest in seen_hashes or prompt_group(row) in blocked:
                continue
            if required is not None:
                if label != "fake" or model != required or model_counts[model] >= required_models[required]:
                    continue
            elif label == "real":
                if real_counts[model] >= real_quotas.get(model, 0):
                    continue
            elif sum(model_counts.values()) >= fake_count or model_counts[model] >= fake_model_cap:
                continue
            chosen.append(row)
            seen_ids.add(row["row_id"])
            seen_hashes.add(digest)
            if label == "real":
                real_counts[model] += 1
            else:
                model_counts[model] += 1
    if (
        dict(real_counts) != real_quotas
        or sum(model_counts.values()) != fake_count
        or any(model_counts[model] < count for model, count in required_models.items())
    ):
        raise ValueError(f"Insufficient quota coverage: real={dict(real_counts)}, fake={dict(model_counts)}")
    return chosen
