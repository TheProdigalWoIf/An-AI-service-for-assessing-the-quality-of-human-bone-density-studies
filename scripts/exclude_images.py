#!/usr/bin/env python3
"""Исключение изображений из обучения БЕЗ удаления файлов.

Ставит exclude_from_training=True в annotations/labels.csv и дописывает строку
в excluded_images/manifest.csv (журнал: что исключено, когда и почему).
DICOM-файлы не перемещаются и не удаляются. Повторный запуск с теми же
аргументами безопасен (идемпотентен).

Примеры:
    python scripts/exclude_images.py --list
    python scripts/exclude_images.py --image-uid 146 --image-uid 147 --reason "эндопротез: нет экспертных критериев"
    python scripts/exclude_images.py --study 70 --reason "исследование целиком"
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

from dxa_qc.repository import LABEL_COLUMNS, LABEL_DELIMITER, LabelStore, _label_to_row

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "annotations" / "labels.csv"
MANIFEST = ROOT / "excluded_images" / "manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Exclude images from training without deleting files")
    parser.add_argument("--image-uid", action="append", default=[], help="image_uid из labels.csv (можно несколько)")
    parser.add_argument("--study", action="append", default=[], help="study_uid целиком (можно несколько)")
    parser.add_argument("--reason", default="", help="Причина исключения (попадает в manifest)")
    args = parser.parse_args()

    store = LabelStore(LABELS)
    labels = store.read()

    if not args.image_uid and not args.study:
        excluded = [item for item in labels if item.exclude_from_training]
        print(f"Excluded from training: {len(excluded)} of {len(labels)}")
        for item in excluded:
            print(f"  image_uid={item.image_uid} study={item.study_uid} path={item.path_to_study} notes={item.notes}")
        return

    if not args.reason:
        raise SystemExit("--reason is required when excluding images")

    selected_uids = set(args.image_uid)
    selected = [
        item for item in labels
        if item.image_uid in selected_uids or item.study_uid in set(args.study)
    ]
    unknown = selected_uids - {item.image_uid for item in labels}
    missing_studies = set(args.study) - {item.study_uid for item in labels}
    if unknown or missing_studies:
        raise SystemExit(f"Unknown image_uids {sorted(unknown)}, unknown studies {sorted(missing_studies)}")
    if not selected:
        raise SystemExit("Nothing matched the requested ids")

    changed = 0
    for item in labels:
        if item in selected and not item.exclude_from_training:
            item.exclude_from_training = True
            changed += 1

    store.write(labels)  # валидирует каждую строку

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    manifest_exists = MANIFEST.exists()
    with MANIFEST.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter=LABEL_DELIMITER)
        if not manifest_exists:
            writer.writerow(LABEL_COLUMNS + ["excluded_at", "reason"])
        for item in selected:
            writer.writerow(
                list(_label_to_row(item).values())
                + [datetime.now(timezone.utc).isoformat(timespec="seconds"), args.reason]
            )

    print(f"Excluded {changed} images ({len(selected)} requested, already excluded: {len(selected) - changed}); "
          f"labels remain {len(labels)}; manifest {MANIFEST}")


if __name__ == "__main__":
    main()
