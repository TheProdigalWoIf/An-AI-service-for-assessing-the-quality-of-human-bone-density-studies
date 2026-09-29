#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dxa_qc.inference import analyze_file
from dxa_qc.repository import dicom_files, studies


COLUMNS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "side",
           "quality_class", "violation_type", "decision_status", "processing_status",
           "time_of_processing", "quality_score", "angle_degrees"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch DXA QC inference")
    parser.add_argument("--data-root", type=Path, default=Path("Results"))
    parser.add_argument("--output", type=Path, default=Path("output/dxa_quality_report.csv"))
    args = parser.parse_args()
    root = args.data_root.resolve()
    rows = [analyze_file(path, root).to_dict() for study in studies(root) for path in dicom_files(study)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row["processing_status"] == "Failure" for row in rows)
    print(f"Processed {len(rows)} images; failures={failures}; report={args.output}")


if __name__ == "__main__":
    main()
