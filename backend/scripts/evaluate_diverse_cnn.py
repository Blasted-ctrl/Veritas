"""Lock calibration thresholds, then compare two CNNs on the same untouched tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from veritas.data.image_corpus import image_identity, load_image_samples
from veritas.models.metadata import file_sha256, metadata_path
from veritas.training.calibration import select_threshold, wilson_interval
from veritas.training.evaluate_image import _group_metrics, evaluate_image
from veritas.training.metrics import compute_metrics


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def bundle_identity(path):
    return {"graph_sha256": file_sha256(path), "processor_sha256": file_sha256(metadata_path(path))}


def evaluate_locked(path, manifest, reference, identity, output):
    if bundle_identity(path) != identity:
        raise ValueError("Model bundle changed after selection/calibration")
    report = evaluate_image(path, manifest, training_dir=reference)
    if bundle_identity(path) != identity:
        raise ValueError("Model bundle changed during evaluation")
    write_json(output, report)
    return report


def summarize(rows, threshold):
    predictions = [
        row | {"threshold": threshold, "prediction": int(row["fake_score"] >= threshold)} for row in rows
    ]
    metrics = compute_metrics(
        [row["label"] for row in rows], [row["fake_score"] for row in rows], threshold=threshold
    ).to_dict()
    matrix = metrics["confusion_matrix"]
    return {
        "metrics": metrics,
        "accuracy_wilson_95": wilson_interval(matrix["tp"] + matrix["tn"], len(rows)),
        "fpr_wilson_95": wilson_interval(matrix["fp"], metrics["n_negative"]),
        "by_source": _group_metrics(predictions, "source"),
        "by_generator": _group_metrics(predictions, "generator_id"),
    }


def verify_roles(directory, source):
    groups, content = set(), set()
    for role, digest in source["manifest_sha256"].items():
        manifest = directory / (role + ".csv")
        if file_sha256(manifest) != digest:
            raise ValueError("Role manifest changed after preparation")
        provenance = json.loads((directory / (role + "-provenance.json")).read_text(encoding="utf-8"))
        samples = load_image_samples(manifest)
        if len(samples) != len(provenance):
            raise ValueError("Role provenance count mismatch")
        local_groups, local_content = set(), set()
        for sample, record in zip(samples, provenance, strict=True):
            raw, pixels = image_identity(sample.path)
            if (raw, pixels) != (record["downloaded_sha256"], record["decoded_sha256"]):
                raise ValueError("Role image changed after preparation")
            if sample.subject_id != record["group"] or sample.subject_id in groups:
                raise ValueError("Role group overlaps another split")
            if {raw, pixels} & (content | local_content):
                raise ValueError("Duplicate content across experiment roles")
            local_groups.add(sample.subject_id)
            local_content.update((raw, pixels))
        groups.update(local_groups)
        content.update(local_content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--baseline-data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "thresholds.json").exists():
        raise ValueError("Use a fresh evaluation directory; thresholds are already locked here")
    selected = json.loads((args.models_dir / "selected.json").read_text(encoding="utf-8"))
    source = json.loads((args.data_dir / "source.json").read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    model = args.models_dir / "selected" / "model.onnx"
    if any(selected[key] != value for key, value in bundle_identity(model).items()):
        raise ValueError("Selected model bundle changed")
    if selected["protocol_sha256"] != file_sha256(args.protocol):
        raise ValueError("Selected protocol changed")
    if selected["dataset_fingerprint"] != source["dataset_fingerprint"]:
        raise ValueError("Selected model data fingerprint differs")
    verify_roles(args.data_dir, source)
    models = {
        "previous_convnext": (args.baseline, args.baseline_data),
        "diverse_efficientnet": (model, args.data_dir),
    }
    identities = {name: bundle_identity(path) for name, (path, _) in models.items()}
    write_json(
        args.output / "selection-lock.json",
        {
            "selected": selected,
            "source_sha256": file_sha256(args.data_dir / "source.json"),
            "models": identities,
            "test_predictions_started": False,
        },
    )
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
            [r["label"] for r in rows],
            [r["fake_score"] for r in rows],
            max_fpr=protocol["calibration"]["target_false_positive_rate"],
        )
    # This file is written before either final test is scored.
    write_json(args.output / "thresholds.json", thresholds)
    summary = {"target": 0.95, "thresholds": thresholds, "cohorts": {}}
    for role in ("id-test", "test"):
        results, original = {}, {}
        for name, (path, reference) in models.items():
            print(f"FINAL {role} {name}", flush=True)
            report = evaluate_locked(
                path,
                args.data_dir / (role + ".csv"),
                reference,
                identities[name],
                args.output / (name + "-" + role + ".json"),
            )
            results[name] = {}
            for variant, detail in report["variants"].items():
                rows = detail["predictions"]
                results[name][variant] = {
                    "fixed_0.5": summarize(rows, 0.5),
                    "calibrated": summarize(rows, thresholds[name]["threshold"]),
                }
                if variant == "original":
                    original[name] = rows
        paired = {}
        old, new = original.values()
        if [r["path"] for r in old] != [r["path"] for r in new]:
            raise ValueError("Paired evaluation examples differ")
        for policy in ("fixed_0.5", "calibrated"):
            old_threshold = 0.5 if policy == "fixed_0.5" else thresholds["previous_convnext"]["threshold"]
            new_threshold = 0.5 if policy == "fixed_0.5" else thresholds["diverse_efficientnet"]["threshold"]
            changes = [
                (int(b["fake_score"] >= new_threshold) == b["label"])
                - (int(a["fake_score"] >= old_threshold) == a["label"])
                for a, b in zip(old, new, strict=True)
            ]
            paired[policy] = {
                "improved": changes.count(1),
                "worsened": changes.count(-1),
                "unchanged": changes.count(0),
            }
        summary["cohorts"][role] = {"models": results, "paired_original": paired}
        write_json(args.output / "summary.json", summary)
    print(
        json.dumps(
            {
                role: result["models"]["diverse_efficientnet"]["original"]["calibrated"]["metrics"]
                for role, result in summary["cohorts"].items()
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
