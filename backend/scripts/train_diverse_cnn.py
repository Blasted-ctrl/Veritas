"""Fit two pre-registered CNN candidates using training and tuning data only."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from veritas.data.image_corpus import audit_image_corpus
from veritas.export.onnx_export import export_image_from_dir
from veritas.models.cnn_model import EfficientNetDetector
from veritas.models.metadata import file_sha256, metadata_path
from veritas.training.cached_cnn import CachedTrainConfig, fit_cached
from veritas.training.seed import seed_everything


def main():
    import torch
    import torchvision

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source = json.loads((args.data_dir / "source.json").read_text(encoding="utf-8"))
    if source["revision"] != protocol["dataset"]["revision"]:
        raise ValueError("Data revision differs from the pre-registered protocol")
    for role, digest in source["manifest_sha256"].items():
        if file_sha256(args.data_dir / (role + ".csv")) != digest:
            raise ValueError("Data role manifest changed after preparation")
    audit = audit_image_corpus(args.data_dir)
    if audit["fingerprint"] != source["dataset_fingerprint"]:
        raise ValueError("Image pixels or manifests changed after preparation")
    torch.set_num_threads(protocol["training"]["threads"])
    results = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, candidate in enumerate(protocol["training"]["candidates"]):
        config = CachedTrainConfig(
            epochs=candidate["epochs"],
            batch_size=protocol["training"]["batch_size"],
            backbone_lr=candidate["backbone_lr"],
            head_lr=candidate["head_lr"],
            real_loss_weight=candidate["real_loss_weight"],
            weight_decay=protocol["training"]["weight_decay"],
            freeze_epochs=protocol["training"]["freeze_epochs"],
            seed=protocol["training"]["seed"],
        )
        seed_everything(config.seed)
        model = EfficientNetDetector(pretrained=True, image_size=224)
        print(f"CANDIDATE {index + 1}: {candidate}", flush=True)
        directory = args.output_dir / f"candidate-{index + 1}"
        result = fit_cached(
            model,
            args.data_dir / "train.csv",
            args.data_dir / "val.csv",
            directory,
            args.cache_dir,
            config,
            data_audit=audit,
        )
        result["candidate"] = index + 1
        results.append(result)
        (args.output_dir / "candidate-results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
        del model
    selected = max(
        results, key=lambda item: (item["best_partial_auc"], item["best_auc"], -item["best_epoch"])
    )
    winner = Path(selected["output_dir"])
    final = args.output_dir / "selected"
    final.mkdir(parents=True, exist_ok=True)
    for filename in ("model.pt", "config.json", "metrics.json", "data-audit.json"):
        shutil.copyfile(winner / filename, final / filename)
    export_image_from_dir(final, final / "model.onnx")
    record = {
        "selection": selected,
        "test_evaluated": False,
        "protocol_sha256": file_sha256(args.protocol),
        "dataset_fingerprint": audit["fingerprint"],
        "checkpoint_sha256": file_sha256(final / "model.pt"),
        "graph_sha256": file_sha256(final / "model.onnx"),
        "processor_sha256": file_sha256(metadata_path(final / "model.onnx")),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "rule": "Select: tuning pAUC@5% FPR, full AUC, earlier epoch; no test predictions",
    }
    (args.output_dir / "selected.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
