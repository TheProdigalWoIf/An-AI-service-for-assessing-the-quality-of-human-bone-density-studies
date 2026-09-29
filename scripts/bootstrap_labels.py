#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from dxa_qc.qc import analyze_file
from dxa_qc.repository import LabelStore, dicom_files, pseudo_label, studies


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate provisional labels for manual review")
    parser.add_argument("--data-root", type=Path, default=Path("Results"))
    parser.add_argument("--output", type=Path, default=Path("annotations/labels.csv"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite {output}; use --overwrite")
    labels = [pseudo_label(analyze_file(path, root)) for study in studies(root) for path in dicom_files(study)]
    LabelStore(output).write(labels)
    print(f"Created {len(labels)} provisional labels at {output}")
    print("All labels have review_status=needs_review and are not clinical ground truth.")


if __name__ == "__main__":
    main()
