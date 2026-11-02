"""Build a bounded, group-separated artwork adaptation set and photo replay set."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import time
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, quote, urlencode, urlparse

from prepare_openfake import download, pace_metadata

from veritas.data.image_corpus import audit_image_corpus, image_identity, load_image_samples
from veritas.data.manifest import Sample, write_manifest
from veritas.models.metadata import file_sha256

SEED = 20261102
SOURCES = {
    "artbench": ("alfredplpl/artbench-pd-256x256", "4056b70025f0e3f916fb271c7a792633ba637484"),
    "lunara": ("moonworks/lunara-art-eval", "5a38a59769a6146e8d1b223c58533d12c60fd7d6"),
}
SIZES = {"train": 256, "val": 32, "calibration": 32, "test": 64}
ILLUSTRATION = re.compile(
    r"anime|manga|cartoon|comic|illustrat|sketch|drawing|line art|pixel art|digital art|hand.drawn",
    re.I,
)
PAINTING = re.compile(r"paint|watercol|oil |acrylic|impressionis|art nouveau|woodblock|folk art", re.I)


def group_role(group):
    value = int(hashlib.sha256(f"{SEED}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 65 else "val" if value < 75 else "calibration" if value < 85 else "test"


def difference_hash(path):
    from PIL import Image

    with Image.open(path) as image:
        values = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).getdata())
    return sum(
        int(values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8)
    )


def cached_json(url, path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pace_metadata()
    result = json.loads(download(url, maximum=16_777_216))
    path.write_text(json.dumps(result), encoding="utf-8")
    return result


def hf_document(url, path):
    document = cached_json(url, path)
    if not any(
        float(parse_qs(urlparse(item["row"]["image"]["src"]).query).get("Expires", ["inf"])[0])
        < time.time() + 300
        for item in document["rows"]
    ):
        return document
    pace_metadata()
    fresh = json.loads(download(url, maximum=16_777_216))

    def identity(page):
        return [
            {
                "index": item["row_idx"],
                "metadata": {key: value for key, value in item["row"].items() if key != "image"},
                "image_path": urlparse(item["row"]["image"]["src"]).path,
            }
            for item in page["rows"]
        ]

    if identity(document) != identity(fresh):
        raise ValueError("Refreshed source identity changed")
    path.write_text(json.dumps(fresh), encoding="utf-8")
    return fresh


def image_url(row, output):
    if row["source"] not in SOURCES:
        return row["url"]
    expiry = parse_qs(urlparse(row["url"]).query).get("Expires", ["inf"])[0]
    if float(expiry) >= time.time() + 300:
        return row["url"]
    kind, index = row["id"].split(":")
    index = int(index)
    offset = index // 100 * 100
    url = "https://datasets-server.huggingface.co/rows?" + urlencode(
        {"dataset": SOURCES[kind][0], "config": "default", "split": "train", "offset": offset, "length": 100}
    )
    document = hf_document(url, output / "metadata" / f"{kind}-{offset}.json")
    return next(item["row"]["image"]["src"] for item in document["rows"] if item["row_idx"] == index)


def hf_rows(output, kind, offsets):
    dataset, revision = SOURCES[kind]
    info = json.loads(download("https://huggingface.co/api/datasets/" + dataset))
    if info["sha"] != revision:
        raise ValueError("Dataset head changed; review the pinned source")
    result = []
    for offset in offsets:
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            {"dataset": dataset, "config": "default", "split": "train", "offset": offset, "length": 100}
        )
        document = hf_document(url, output / "metadata" / f"{kind}-{offset}.json")
        for item in document["rows"]:
            row = item["row"]
            if f"/--/{revision}/--/" not in row["image"]["src"]:
                raise ValueError("Viewer revision differs from pinned source")
            index = item["row_idx"]
            if kind == "artbench":
                group = "artist:" + row["artist"]
                domain, label, generator = "painting", 0, ""
                origin, license_name = (
                    row["url"],
                    "publisher-declared public-domain artwork; MIT dataset metadata",
                )
            else:
                prompt = " ".join(row["prompt"].casefold().split())
                domain = (
                    "illustration"
                    if ILLUSTRATION.search(prompt)
                    else "painting"
                    if PAINTING.search(prompt)
                    else ""
                )
                if not domain:
                    continue
                group = "prompt:" + hashlib.sha256(prompt.encode()).hexdigest()
                label, generator = 1, row["model"]
                origin, license_name = (
                    f"https://huggingface.co/datasets/{dataset}/viewer/default/train?row={index}",
                    "Apache-2.0 dataset declaration",
                )
            result.append(
                {
                    "id": f"{kind}:{index}",
                    "label": label,
                    "domain": domain,
                    "group": group,
                    "source": kind,
                    "generator": generator,
                    "url": row["image"]["src"],
                    "origin": origin,
                    "license": license_name,
                }
            )
        print(f"METADATA {kind} {offset}", flush=True)
    return result


def check_anime_version():
    info = json.loads(
        download("https://www.kaggle.com/api/v1/datasets/view/stevenevan99/pixiv-top-daily-illustration-2018")
    )
    if info["currentVersionNumber"] != 1:
        raise ValueError("Pixiv source version changed")
    return info


def anime_rows(output):
    identifier = "stevenevan99/pixiv-top-daily-illustration-2018"
    info = check_anime_version()
    (output / "metadata" / "pixiv-info.json").write_text(json.dumps(info), encoding="utf-8")
    token, result = "", []
    for page in range(12):
        query = {"pageSize": 100}
        if token:
            query["pageToken"] = token
        document = cached_json(
            "https://www.kaggle.com/api/v1/datasets/list/" + identifier + "?" + urlencode(query),
            output / "metadata" / f"pixiv-{page}.json",
        )
        for row in document["datasetFiles"]:
            name = row["name"]
            match = re.fullmatch(r"2018/(\d+)_p\d+_master1200\.(jpg|png)", name)
            if not match:
                continue
            result.append(
                {
                    "id": "pixiv:" + name,
                    "label": 0,
                    "domain": "illustration",
                    "group": "pixiv-work:" + match[1],
                    "source": "pixiv2018",
                    "generator": "",
                    "url": "https://www.kaggle.com/api/v1/datasets/download/"
                    + identifier
                    + "/"
                    + quote(name, safe=""),
                    "origin": "https://www.pixiv.net/artworks/" + match[1],
                    "license": "Copyright retained by original artists; dataset collected January 2021",
                }
            )
        token = document.get("nextPageToken", "")
        if not token:
            break
    return result


def prepare(output, replay):
    from PIL import Image

    output.mkdir(parents=True, exist_ok=True)
    (output / "metadata").mkdir(exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    candidates = anime_rows(output)
    candidates += hf_rows(output, "artbench", random.Random(SEED).sample(range(0, 46200, 100), 32))
    candidates += hf_rows(output, "lunara", range(0, 8000, 100))
    candidates.sort(key=lambda row: hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest())
    roles = {role: [] for role in SIZES}
    counts, group_counts, generator_counts = Counter(), Counter(), Counter()
    fingerprints, rejected, provenance, perceptual = set(), [], [], []
    for row in candidates:
        role = group_role(row["group"])
        key = (role, row["label"], row["domain"])
        if counts[key] >= SIZES[role]:
            continue
        if row["source"] == "artbench" and group_counts[row["group"]] >= 16:
            continue
        if row["label"] and generator_counts[(role, row["generator"])] >= max(16, SIZES[role] // 2):
            continue
        destination = output / "images" / (hashlib.sha256(row["id"].encode()).hexdigest() + ".jpg")
        original = destination.with_suffix(".original")
        url = image_url(row, output) if not original.exists() else None
        try:
            if not original.exists():
                original.write_bytes(download(url, maximum=8_388_608))
            raw = original.read_bytes()
            if not destination.exists():
                with Image.open(io.BytesIO(raw)) as image:
                    image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR).save(
                        destination, "JPEG", quality=95, subsampling=0
                    )
            encoded, decoded = image_identity(destination)
            if {encoded, decoded} & fingerprints:
                continue
            dhash = difference_hash(destination)
            if any(
                previous_role != role and (dhash ^ previous).bit_count() <= 4
                for previous, previous_role in perceptual
            ):
                rejected.append({"id": row["id"], "error": "Cross-role near duplicate"})
                continue
            fingerprints.update((encoded, decoded))
            perceptual.append((dhash, role))
        except (OSError, ValueError) as error:
            rejected.append({"id": row["id"], "error": str(error)})
            continue
        split = role if role in ("train", "val") else "test"
        roles[role].append(
            Sample(
                str(destination.resolve()),
                row["label"],
                row["group"],
                "image",
                row["source"] + ":" + row["domain"],
                split,
                row["generator"],
                row["origin"],
                row["license"],
            )
        )
        provenance.append(
            {k: v for k, v in row.items() if k != "url"}
            | {
                "role": role,
                "downloaded_sha256": hashlib.sha256(raw).hexdigest(),
                "prepared_sha256": encoded,
                "decoded_sha256": decoded,
            }
        )
        counts[key] += 1
        group_counts[row["group"]] += 1
        generator_counts[(role, row["generator"])] += 1
        if len(provenance) % 64 == 0:
            print(f"ARTWORK acquired {len(provenance)}/1536", flush=True)
    for role, count in SIZES.items():
        for label in (0, 1):
            for domain in ("illustration", "painting"):
                if counts[(role, label, domain)] != count:
                    raise ValueError(
                        f"Insufficient style/class quota: {role}/{label}/{domain}: "
                        f"{counts[(role, label, domain)]}/{count}"
                    )
    for role, source_role, per_class in [
        ("train", "train", 256),
        ("val", "val", 64),
        ("calibration", "calibration", 64),
        ("test", "id-test", 128),
    ]:
        pool = load_image_samples(replay / (source_role + ".csv"))
        for label in (0, 1):
            chosen = [s for s in pool if s.label == label]
            random.Random(SEED).shuffle(chosen)
            accepted = 0
            for sample in chosen:
                if accepted == per_class:
                    break
                destination = (
                    output / "images" / ("photo-" + hashlib.sha256(sample.path.encode()).hexdigest() + ".jpg")
                )
                with Image.open(sample.path) as image:
                    image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR).save(
                        destination, "JPEG", quality=95, subsampling=0
                    )
                encoded, decoded = image_identity(destination)
                if {encoded, decoded} & fingerprints:
                    continue
                dhash = difference_hash(destination)
                if any(
                    previous_role != role and (dhash ^ previous).bit_count() <= 4
                    for previous, previous_role in perceptual
                ):
                    continue
                perceptual.append((dhash, role))
                accepted += 1
                fingerprints.update((encoded, decoded))
                roles[role].append(
                    Sample(
                        str(destination.resolve()),
                        label,
                        sample.subject_id,
                        "image",
                        "photo-replay:" + sample.source,
                        role if role in ("train", "val") else "test",
                        sample.generator_id,
                        sample.source_url,
                        sample.license,
                    )
                )
                provenance.append(
                    {
                        "id": "photo:" + sample.path,
                        "label": label,
                        "domain": "photo",
                        "group": sample.subject_id,
                        "source": sample.source,
                        "role": role,
                        "downloaded_sha256": file_sha256(sample.path),
                        "prepared_sha256": encoded,
                        "decoded_sha256": decoded,
                    }
                )
            if accepted != per_class:
                raise ValueError("Insufficient disjoint photo replay")
    check_anime_version()
    for role, rows in roles.items():
        write_manifest(rows, output / (role + ".csv"))
    audit = audit_image_corpus(output)
    groups = {}
    for row in provenance:
        if groups.setdefault(row["group"], row["role"]) != row["role"]:
            raise ValueError("Source group crosses experimental roles")
    source = {
        "revision": "artwork-v3",
        "seed": SEED,
        "sources": SOURCES,
        "pixiv_version": 1,
        "dataset_fingerprint": audit["fingerprint"],
        "manifest_sha256": {role: file_sha256(output / (role + ".csv")) for role in roles},
        "counts": {role: len(rows) for role, rows in roles.items()},
        "artwork_quotas_per_class_and_style": SIZES,
        "limits": [
            "Curator origin labels are not independently authenticated.",
            "Pixiv grouping is by artwork; artist-disjoint evaluation cannot be established.",
            "ArtBench artist groups and AI prompt groups remain within one role.",
            "Photo test is a previously published regression cohort, not a new sealed test.",
            "All examples use 256px JPEG95 preparation, not arbitrary resolutions.",
            "Original image rights remain with their respective owners; no raw images are committed.",
        ],
    }
    for name, value in [
        ("source.json", source),
        ("provenance.json", provenance),
        ("data-audit.json", audit),
        ("rejected.json", rejected),
    ]:
        (output / name).write_text(json.dumps(value, indent=2), encoding="utf-8")
    print(json.dumps(source, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--photo-replay", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.output, arguments.photo_replay)
