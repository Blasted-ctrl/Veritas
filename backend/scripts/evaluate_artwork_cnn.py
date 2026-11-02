"""Calibrate three fixed models, then score the same held-out artwork and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluate_diverse_cnn import bundle_identity, evaluate_locked, summarize, write_json

from veritas.data.image_corpus import image_identity, load_image_samples
from veritas.models.metadata import file_sha256
from veritas.training.calibration import select_threshold


def verify_roles(directory, source, diagnostic_hash):
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    records = {row["prepared_sha256"]: row for row in provenance}
    if len(records) != len(provenance):
        raise ValueError("Duplicate provenance content")
    if any(diagnostic_hash in (row["downloaded_sha256"], row["prepared_sha256"]) for row in provenance):
        raise ValueError("Diagnostic image leaked into experiment")
    groups, content, observed = {}, set(), set()
    for role, digest in source["manifest_sha256"].items():
        manifest = directory / (role + ".csv")
        if file_sha256(manifest) != digest:
            raise ValueError("Role manifest changed after preparation")
        for row in load_image_samples(manifest):
            encoded, decoded = image_identity(row.path)
            record = records[encoded]
            if (
                record["decoded_sha256"] != decoded
                or record["role"] != role
                or record["label"] != row.label
                or record["group"] != row.subject_id
            ):
                raise ValueError("Manifest/provenance identity differs")
            if {encoded, decoded} & content:
                raise ValueError("Duplicate experimental content")
            if groups.setdefault(row.subject_id, role) != role:
                raise ValueError("Group crosses experimental roles")
            content.update((encoded, decoded))
            observed.add(encoded)
    if observed != set(records):
        raise ValueError("Unaccounted provenance rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--previous-model", type=Path, required=True)
    parser.add_argument("--previous-data", type=Path, required=True)
    parser.add_argument("--default-model", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibrate", "test"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = json.loads((args.data_dir / "source.json").read_text(encoding="utf-8"))
    selected = json.loads((args.models_dir / "selected.json").read_text(encoding="utf-8"))
    model = args.models_dir / "selected" / "model.onnx"
    if selected["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("Selected protocol changed")
    if selected["dataset_fingerprint"] != source["dataset_fingerprint"]:
        raise ValueError("Selected data fingerprint changed")
    if any(selected[key] != value for key, value in bundle_identity(model).items()):
        raise ValueError("Selected model bundle changed")
    verify_roles(args.data_dir, source, protocol["evaluation"]["diagnostic"]["file_sha256"])
    models = {
        "default_vit": (args.default_model, None),
        "previous_cnn": (args.previous_model, args.previous_data),
        "artwork_cnn": (model, args.data_dir),
    }
    identities = {name: bundle_identity(path) for name, (path, _) in models.items()}
    identity = {
        "protocol_sha256": file_sha256(args.protocol),
        "source_sha256": file_sha256(args.data_dir / "source.json"),
        "models": identities,
        "selected": selected,
    }
    lock_path = args.output / "thresholds.json"
    if args.phase == "calibrate":
        if lock_path.exists():
            raise ValueError("Thresholds already locked; use a fresh experiment")
        thresholds = {}
        for name, (path, reference) in models.items():
            print(f"CALIBRATION {name}", flush=True)
            report = evaluate_locked(
                path,
                args.data_dir / "calibration.csv",
                reference,
                identities[name],
                args.output / (name + "-calibration.json"),
            )
            rows = report["variants"]["original"]["predictions"]
            thresholds[name] = select_threshold(
                [row["label"] for row in rows],
                [row["fake_score"] for row in rows],
                max_fpr=0.05,
            )
        write_json(lock_path, identity | {"thresholds": thresholds, "test_predictions_started": False})
        print("Thresholds locked before any final test inference", flush=True)
        return
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if any(lock[key] != value for key, value in identity.items()):
        raise ValueError("Experiment changed after calibration")
    if (args.output / "test-started.json").exists():
        raise ValueError("This test was already started; preserve its results")
    write_json(args.output / "test-started.json", {"threshold_lock_sha256": file_sha256(lock_path)})
    results, paths = {}, None
    for name, (path, reference) in models.items():
        print(f"TEST {name}", flush=True)
        report = evaluate_locked(
            path,
            args.data_dir / "test.csv",
            reference,
            identities[name],
            args.output / (name + "-test.json"),
        )
        current_paths = [row["path"] for row in report["variants"]["original"]["predictions"]]
        if paths is not None and paths != current_paths:
            raise ValueError("Paired test examples differ")
        paths = current_paths
        results[name] = {}
        for variant, detail in report["variants"].items():
            rows = detail["predictions"]
            cohorts = {
                "all": rows,
                "artwork": [row for row in rows if not row["source"].startswith("photo-replay:")],
                "general_regression": [row for row in rows if row["source"].startswith("photo-replay:")],
            }
            results[name][variant] = {
                cohort: {
                    "fixed_0.5": summarize(members, 0.5),
                    "calibrated": summarize(members, lock["thresholds"][name]["threshold"]),
                }
                for cohort, members in cohorts.items()
            }
        write_json(args.output / "summary.json", {"models": results})
    print("All three fixed models scored on identical examples", flush=True)


if __name__ == "__main__":
    main()
