"""Download a bounded CIFAKE pilot from its author's pinned public Git tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from veritas.data.image_corpus import audit_image_corpus
from veritas.data.manifest import Sample, write_manifest

REPOSITORY = "jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images"
REVISION = "e112a942abaecd02b6b1f6f646c807d56be8fb62"
RAW = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/"


def download(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Veritas-CIFAKE-pilot"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=64)
    parser.add_argument("--val-per-class", type=int, default=16)
    parser.add_argument("--test-per-class", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    if min(args.train_per_class, args.val_per_class, args.test_per_class) < 1:
        parser.error("Each split needs at least one example of each class")
    args.output.mkdir(parents=True, exist_ok=True)
    selected = []
    listings = {}
    for label, name in enumerate(("REAL", "FAKE")):
        for official_split in ("train", "test"):
            folder = f"DATASET/{official_split}/{name}"
            url = f"https://api.github.com/repos/{REPOSITORY}/contents/{folder}?ref={REVISION}"
            entries = [
                row
                for row in json.loads(download(url))
                if row["type"] == "file" and row["name"].lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            entries.sort(key=lambda row: row["path"])
            listings[folder] = len(entries)
            random.Random(args.seed + label + (10 if official_split == "test" else 0)).shuffle(entries)
            needed = (
                args.test_per_class if official_split == "test" else args.train_per_class + args.val_per_class
            )
            if needed > len(entries):
                parser.error(f"Pilot request exceeds the bounded GitHub listing for {folder}")
            for index, entry in enumerate(entries[:needed]):
                split = (
                    "test" if official_split == "test" else "train" if index < args.train_per_class else "val"
                )
                selected.append((split, label, entry))

    def materialize(item):
        split, label, entry = item
        blob = entry["sha"]
        if len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob):
            raise ValueError("Invalid Git blob identity")
        path = args.output / "images" / (blob + Path(entry["name"]).suffix.lower())
        url = RAW + quote(entry["path"], safe="/")
        payload = path.read_bytes() if path.exists() else download(url)
        actual = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
        if actual != blob:
            raise ValueError(f"Downloaded image differs from pinned blob {blob}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
        return Sample(
            path=str(path.resolve()),
            label=label,
            subject_id="cifake:" + blob,
            modality="image",
            source="cifake-pilot",
            split=split,
            generator_id="stable-diffusion-v1.4" if label else "",
            source_url=url,
            license="MIT",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        samples = list(pool.map(materialize, selected))
    for split in ("train", "val", "test"):
        write_manifest(
            [sample for sample in samples if sample.split == split], args.output / (split + ".csv")
        )
    source_readme = download(RAW + "README.md")
    (args.output / "SOURCE-README.md").write_bytes(source_readme)
    provenance = {
        "role": "bounded public-data pilot; not a modern-generator accuracy benchmark",
        "source_repository": REPOSITORY,
        "source_revision": REVISION,
        "seed": args.seed,
        "listed_files_per_folder": listings,
        "samples": len(samples),
        "selection": "Seeded selection from bounded GitHub directory listings; official test stays test",
        "license": "MIT, as declared by the dataset author; source README retained",
        "limits": [
            "32x32 inputs",
            "One old generator",
            "Content-group IDs do not establish human identity",
            "Not a uniform sample of the entire dataset",
            "No broad real-world accuracy claim",
        ],
    }
    (args.output / "pilot.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    audit = audit_image_corpus(args.output)
    (args.output / "data-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
