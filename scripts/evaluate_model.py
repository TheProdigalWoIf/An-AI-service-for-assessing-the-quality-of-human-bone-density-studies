#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from torch.utils.data import DataLoader

from dxa_qc.ml import choose_device, load_checkpoint
from dxa_qc.repository import LabelStore
from train_model_2 import DXADataset, build_metrics, predict


def main():
    parser = argparse.ArgumentParser(description="Evaluate the saved DXA model on the validation split")
    parser.add_argument("--data-root", type=Path, default=Path("Results"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/labels.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    with (args.model_dir / "splits.csv").open(encoding="utf-8-sig", newline="") as stream:
        split_uids = {row["image_uid"] for row in csv.DictReader(stream) if row["split"] == "validation"}
    rows = [row for row in LabelStore(args.labels).read() if row.image_uid in split_uids]
    device = choose_device()
    model, checkpoint = load_checkpoint(args.model_dir / "dxa_qc_resnet18.pt", device)
    result = predict(model, DataLoader(DXADataset(rows, args.data_root.resolve(), False), batch_size=24), device)
    metrics_path = args.model_dir / "metrics.json"
    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    document["validation"] = build_metrics(result, checkpoint["quality_threshold"], checkpoint["violation_thresholds"])
    metrics_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(document["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
