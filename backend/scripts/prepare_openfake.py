"""Acquire a bounded, revision-checked OpenFake research subset without multi-TB parquet downloads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import BoundedSemaphore, Lock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

from veritas.data.diverse import prompt_group, select_rows
from veritas.data.image_corpus import audit_image_corpus, image_identity
from veritas.data.manifest import Sample, write_manifest
from veritas.models.metadata import file_sha256

REPOSITORY = "ComplexDataLab/OpenFake"
REVISION = "3247b942f18d39d1fd476b1077852af033d35d8c"
SEED = 20261101
SIZES = {"train": 2300034, "validation": 58453, "test": 91397}
PAGES = {"train": 96, "validation": 64, "test": 40}
DOWNLOAD_ATTEMPTS = 8


_api_lock = Lock()
_api_not_before = 0.0


def pace_metadata():
    global _api_not_before
    while True:
        with _api_lock:
            wait = _api_not_before - time.monotonic()
            if wait <= 0:
                _api_not_before = time.monotonic() + 5.0
                return
        time.sleep(min(wait, 60))


def retry_delay(error, attempt):
    value = error.headers.get("Retry-After")
    if value:
        try:
            if value.isdecimal():
                return max(float(value), 2**attempt)
            target = parsedate_to_datetime(value).timestamp()
            server_date = error.headers.get("Date")
            reference = parsedate_to_datetime(server_date).timestamp() if server_date else time.time()
            return max(target - reference, 2**attempt)
        except (ValueError, TypeError, OverflowError):
            pass
    return max(60 if error.code == 429 else 30, 2**attempt)


def download(url, maximum=4_194_304):
    global _api_not_before
    for attempt in range(DOWNLOAD_ATTEMPTS):
        try:
            request = Request(url, headers={"User-Agent": "Veritas-research/0.2"})
            with urlopen(request, timeout=30) as response:
                data = response.read(maximum + 1)
            if len(data) > maximum:
                raise ValueError("Response exceeds bounded download size")
            return data
        except HTTPError as error:
            if attempt == DOWNLOAD_ATTEMPTS - 1 or error.code not in (429, 500, 502, 503, 504):
                raise
            delay = retry_delay(error, attempt)
            with _api_lock:
                _api_not_before = max(_api_not_before, time.monotonic() + delay)
            print(f"HTTP {error.code} from {urlparse(url).hostname}: retry delay {delay:.0f}s", flush=True)
            time.sleep(delay)
        except (OSError, TimeoutError):
            if attempt == DOWNLOAD_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable download state")


def viewer_rows(document, split):
    rows = []
    for item in document.get("rows", []):
        row = item["row"]
        parsed = urlparse((row.get("image") or {}).get("src", ""))
        marker = f"/--/{REVISION}/--/core/{split}/"
        if parsed.hostname != "datasets-server.huggingface.co" or marker not in parsed.path:
            raise ValueError("Viewer image revision does not match pinned source")
        index = int(parsed.path.split(marker, 1)[1].split("/", 1)[0])
        if not 0 <= index < SIZES[split]:
            raise ValueError("Invalid upstream row index")
        rows.append(
            row | {"model": row["model"].strip().casefold(), "upstream_split": split, "row_idx": index}
        )
    if not rows:
        raise ValueError("Dataset viewer returned no rows")
    return rows


class ViewerLinks:
    def __init__(self):
        self.pages = {}
        self.locks = {}
        self.guard = Lock()
        self.requests = BoundedSemaphore(3)

    def source(self, row):
        key = (row["upstream_split"], row["row_idx"] // 100 * 100)
        with self.guard:
            lock = self.locks.setdefault(key, Lock())
        with lock:
            page = self.pages.get(key, [])
            fresh = next((item for item in page if item["row_id"] == row["row_id"]), None)
            expiry = (
                parse_qs(urlparse(fresh["image"]["src"]).query).get("Expires", ["0"])[0] if fresh else "0"
            )
            if float(expiry) < time.time() + 300:
                url = "https://datasets-server.huggingface.co/rows?" + urlencode(
                    {
                        "dataset": REPOSITORY,
                        "config": "core",
                        "split": key[0],
                        "offset": key[1],
                        "length": 100,
                    }
                )
                with self.requests:
                    pace_metadata()
                    self.pages[key] = viewer_rows(json.loads(download(url)), key[0])
            fresh = next(item for item in self.pages[key] if item["row_id"] == row["row_id"])
            if fresh["image_sha256"] != row["image_sha256"] or prompt_group(fresh) != prompt_group(row):
                raise ValueError("Refreshed source identity changed")
            return fresh["image"]["src"]


def check_revision():
    info = json.loads(download("https://huggingface.co/api/datasets/" + REPOSITORY))
    if info["sha"] != REVISION:
        raise ValueError("Dataset head changed; review/pin a new experiment before collecting")


def metadata(output):
    check_revision()
    directory = output / "metadata"
    directory.mkdir(parents=True, exist_ok=True)
    jobs = []
    for number, (split, size) in enumerate(SIZES.items()):
        offsets = random.Random(SEED + number).sample(range(size // 100), PAGES[split])
        jobs.extend((split, offset * 100) for offset in offsets)

    original = {offset for split, offset in jobs if split == "train"}
    extra = random.Random(SEED + 1000).sample(
        [offset * 100 for offset in range(SIZES["train"] // 100) if offset * 100 not in original], 64
    )
    jobs.extend(("train", offset) for offset in extra)

    def page(job):
        split, offset = job
        path = directory / f"{split}-{offset}.json"
        url = "https://datasets-server.huggingface.co/rows?" + urlencode(
            {"dataset": REPOSITORY, "config": "core", "split": split, "offset": offset, "length": 100}
        )
        if path.exists():
            document = json.loads(path.read_text(encoding="utf-8"))
        else:
            pace_metadata()
            document = json.loads(download(url))
        rows = viewer_rows(document, split)
        if not path.exists():
            temporary = path.with_suffix(".json.part")
            temporary.write_text(json.dumps(document), encoding="utf-8")
            temporary.replace(path)
        return split, rows

    pools = {split: [] for split in SIZES}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for number, (split, rows) in enumerate(pool.map(page, jobs), 1):
            pools[split].extend(rows)
            if number % 10 == 0:
                print(f"Metadata pages {number}/{len(jobs)}", flush=True)
    check_revision()
    for split, rows in pools.items():
        (output / f"pool-{split}.json").write_text(json.dumps(rows), encoding="utf-8")
        print(split, dict(Counter(row["label"] + ":" + row["model"] for row in rows)), flush=True)
    return pools


def prepare(output, pools):
    from PIL import Image

    pools = {
        split: [row | {"model": row["model"].strip().casefold()} for row in rows]
        for split, rows in pools.items()
    }
    heldout_models = set()
    fresh_source = ViewerLinks().source

    roles = [
        ("test", "test", {"docci": 256, "imagenet": 256}, 512, 128),
        ("id-test", "validation", {"laion": 128, "pexels": 128}, 256, 64),
        ("calibration", "validation", {"laion": 128, "pexels": 128}, 256, 64),
        ("val", "validation", {"laion": 128, "pexels": 128}, 256, 64),
        ("train", "train", {"laion": 1024, "pexels": 1024}, 2048, 256),
    ]
    groups, original_hashes = set(), set()
    encoded, decoded = {}, {}
    perceptual = []
    rejected = []
    summaries = {}
    for role, upstream, quotas, fakes, cap in roles:
        candidates = [
            row
            for row in pools[upstream]
            if row["label"] == "real"
            or (not row["model"].startswith(("sd-1.", "sd-2.")) and row["model"] not in heldout_models)
        ]
        required = {"flux.2-dev": 32, "qwen-image": 32, "sd-3.5": 32} if role == "train" else {}
        # Rejected duplicates are removed before refilling the same fixed quotas.
        while True:
            chosen = select_rows(
                candidates,
                real_quotas=quotas,
                fake_count=fakes,
                fake_model_cap=cap,
                seed=SEED,
                excluded_groups=groups,
                excluded_hashes=original_hashes,
                required_models=required,
            )

            def materialize(row):
                name = (
                    hashlib.sha256((REVISION + row["upstream_split"] + row["row_id"]).encode()).hexdigest()
                    + ".jpg"
                )
                path = output / "images" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    raw = path.read_bytes()
                else:
                    url = row["image"]["src"]
                    expiry = parse_qs(urlparse(url).query).get("Expires", ["0"])[0]
                    if float(expiry) < time.time() + 300:
                        url = fresh_source(row)
                    try:
                        raw = download(url, 16_777_216)
                    except HTTPError as error:
                        if error.code != 403:
                            raise
                        raw = download(fresh_source(row), 16_777_216)
                with Image.open(io.BytesIO(raw)) as image:
                    image.verify()
                if not path.exists():
                    temporary = path.with_suffix(".part")
                    temporary.write_bytes(raw)
                    temporary.replace(path)
                raw_hash, pixels = image_identity(path)
                with Image.open(path) as image:
                    thumb = image.convert("L").resize((9, 8))
                    values = list(thumb.getdata())
                    dhash = sum(
                        (values[y * 9 + x] > values[y * 9 + x + 1]) << (y * 8 + x)
                        for y in range(8)
                        for x in range(8)
                    )
                return row, path, raw_hash, pixels, dhash

            accepted = []
            removed = set()
            with ThreadPoolExecutor(max_workers=6) as pool:
                for index, item in enumerate(pool.map(materialize, chosen), 1):
                    row, path, raw_hash, pixels, dhash = item
                    prior = encoded.get(raw_hash) or decoded.get(pixels)
                    if prior and prior[1] != (1 if row["label"] == "fake" else 0):
                        raise ValueError("Conflicting labels for identical image content")
                    near = any(
                        previous_role != role and (dhash ^ prior_hash).bit_count() <= 4
                        for prior_hash, previous_role in perceptual
                    )
                    if prior or near:
                        removed.add(row["row_id"])
                        rejected.append(
                            {"role": role, "row_id": row["row_id"], "reason": "content/near-duplicate"}
                        )
                        continue
                    accepted.append(item)
                    if index % 128 == 0:
                        print(f"{role}: downloaded/checked {index}/{len(chosen)}", flush=True)
            # Resolve duplicates among this role without changing higher-priority sealed roles.
            local_encoded, local_pixels = {}, {}
            for row, _path, raw_hash, pixels, _dhash in accepted:
                previous = local_encoded.get(raw_hash) or local_pixels.get(pixels)
                if previous:
                    if previous["label"] != row["label"]:
                        raise ValueError("Conflicting labels for identical image content")
                    removed.add(row["row_id"])
                local_encoded[raw_hash] = row
                local_pixels[pixels] = row
            if removed:
                candidates = [row for row in candidates if row["row_id"] not in removed]
                continue
            manifests = []
            provenance = []
            for row, path, raw_hash, pixels, dhash in accepted:
                label = 1 if row["label"] == "fake" else 0
                groups.add(prompt_group(row))
                original_hashes.add(row["image_sha256"])
                encoded[raw_hash] = (role, label)
                decoded[pixels] = (role, label)
                perceptual.append((dhash, role))
                source_url = "https://datasets-server.huggingface.co/rows?" + urlencode(
                    {
                        "dataset": REPOSITORY,
                        "config": "core",
                        "split": upstream,
                        "offset": row["row_idx"],
                        "length": 1,
                    }
                )
                manifests.append(
                    Sample(
                        str(path.resolve()),
                        label,
                        prompt_group(row),
                        "image",
                        "openfake:" + row["model"] if label == 0 else "openfake",
                        role if role in ("train", "val", "test") else "test",
                        row["model"] if label else "",
                        source_url,
                        "CC-BY-NC-4.0",
                    )
                )
                provenance.append(
                    {
                        key: row[key]
                        for key in (
                            "row_id",
                            "row_idx",
                            "upstream_split",
                            "label",
                            "model",
                            "image_sha256",
                            "prompt",
                        )
                    }
                    | {
                        "downloaded_sha256": raw_hash,
                        "decoded_sha256": pixels,
                        "dhash": f"{dhash:016x}",
                        "group": prompt_group(row),
                        "viewer_derivative": True,
                        "revision": REVISION,
                    }
                )
            write_manifest(manifests, output / (role + ".csv"))
            (output / (role + "-provenance.json")).write_text(
                json.dumps(provenance, indent=2), encoding="utf-8"
            )
            summaries[role] = {
                "samples": len(manifests),
                "labels": dict(Counter(row.label for row in manifests)),
                "models": dict(Counter(row["model"] for row, _, _, _, _ in accepted)),
            }
            if role == "test":
                heldout_models = {row["model"] for row, *_ in accepted if row["label"] == "fake"}
            print(role, summaries[role], flush=True)
            break
    audit = audit_image_corpus(output, heldout_generators=sorted(heldout_models))
    (output / "data-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    (output / "source.json").write_text(
        json.dumps(
            {
                "repository": REPOSITORY,
                "revision": REVISION,
                "seed": SEED,
                "license": "CC-BY-NC-4.0 research dataset; source rights also apply",
                "derivatives": "Viewer JPEGs; original and downloaded hashes recorded separately",
                "selection": "Seeded noncontiguous pages; fixed source/model quotas, not uniform sampling",
                "roles": summaries,
                "manifest_sha256": {role: file_sha256(output / (role + ".csv")) for role in summaries},
                "dataset_fingerprint": audit["fingerprint"],
                "rejected": rejected,
                "near_duplicate_check": "Cross-role 64-bit dHash distance<=4; not an identity guarantee",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=["metadata", "prepare"], default="prepare")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    pools = metadata(args.output)
    if args.stage == "prepare":
        prepare(args.output, pools)


if __name__ == "__main__":
    main()
