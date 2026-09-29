from __future__ import annotations

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from dxa_qc.models import LabelRecord, QCResult


# Схема labels.csv (разделитель ';' — как в экспериментальной разметке Excel).
LABEL_COLUMNS = [
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "side",
    "quality_class",
    "correct_positioning",
    "spine_axis_alignment",
    "foreign_objects_artifacts",
    "hip_positioning_rotation",
    "hip_roi_correctness",
    "notes",
    "exclude_from_training",
]
LABEL_DELIMITER = ";"

# Эксперт пишет в Excel «hip»; внутри системы область canonical — proximal_femur.
REGION_ALIASES = {
    "lumbar_spine": "lumbar_spine",
    "proximal_femur": "proximal_femur",
    "hip": "proximal_femur",
    "unknown": "unknown",
}
VALID_REGIONS = set(REGION_ALIASES)
VALID_SIDES = {"none", "right", "left"}

# Колонка критерия в CSV <-> код нарушения в модели.
# Значение 1 в колонке = критерий нарушен.
CRITERION_BY_VIOLATION = {
    "spine_positioning_incorrect": "correct_positioning",
    "spine_axis_misalignment": "spine_axis_alignment",
    "spine_artifact_or_object": "foreign_objects_artifacts",
    "hip_positioning_incorrect": "hip_positioning_rotation",
    "hip_roi_incorrect": "hip_roi_correctness",
}
VIOLATION_BY_CRITERION = {column: code for code, column in CRITERION_BY_VIOLATION.items()}
VALID_VIOLATIONS = set(CRITERION_BY_VIOLATION)


def allowed_violation_codes(region: str) -> set[str]:
    """Какие коды нарушений применимы к области (единый словарь с dxa_qc.ml)."""
    from dxa_qc.ml import allowed_violations

    return allowed_violations(REGION_ALIASES.get(region, region))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def studies(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted(path for path in data_root.iterdir() if path.is_dir() and any(path.rglob("*.dcm")))


def dicom_files(study: Path) -> list[Path]:
    return sorted(study.rglob("*.dcm"))


def violation_codes(label: LabelRecord) -> list[str]:
    if not label.violation_type or label.violation_type.strip() == "none":
        return []
    return [item.strip() for item in label.violation_type.split(";") if item.strip() and item.strip() != "none"]


def validate_label(label: LabelRecord) -> None:
    region = label.anatomical_region
    if region not in VALID_REGIONS:
        raise ValueError("Unsupported anatomical_region")
    region = REGION_ALIASES[region]

    if label.side not in VALID_SIDES:
        raise ValueError("Unsupported side")

    if region == "lumbar_spine" and label.side != "none":
        raise ValueError("Spine must have side='none'")
    if region == "proximal_femur" and label.side not in {"right", "left"}:
        raise ValueError("Femur must have side='right' or side='left'")

    if label.quality_class not in {0, 1, None}:
        raise ValueError("quality_class must be 0, 1 or empty")

    selected = set(violation_codes(label))
    unknown = selected - VALID_VIOLATIONS
    if unknown:
        raise ValueError(f"Unsupported violation_type: {sorted(unknown)}")
    not_allowed = selected - allowed_violation_codes(region)
    if not_allowed:
        raise ValueError(f"Violations {sorted(not_allowed)} are not applicable to {region}")

    # Размеченная строка должна быть самосогласована: плохой снимок (1) имеет
    # хотя бы один нарушенный критерий, хороший (0) — ни одного.
    if label.quality_class is not None:
        if label.quality_class == 0 and selected:
            raise ValueError("quality_class=0 requires no violated criteria")
        if label.quality_class == 1 and not selected:
            raise ValueError("quality_class=1 requires at least one violated criterion")


def pseudo_label(result: QCResult) -> LabelRecord:
    """Псевдометка от правил qc.py.

    Коды правил (low_contrast и т.п.) не входят в пять экспертных критериев,
    поэтому строка создаётся неразмеченной (quality_class=None): до ручной
    проверки эксперта она в обучение не попадает. Подсказка правил — в notes.
    """
    hint = f"bootstrap_rules: {result.violation_type}"
    return LabelRecord(
        result.path_to_study,
        result.study_uid,
        result.image_uid,
        REGION_ALIASES.get(result.anatomical_region, result.anatomical_region),
        result.side if result.side in VALID_SIDES else "none",
        None,
        "none",
        hint,
    )


def _read_int_or_none(value: str | None) -> int | None:
    value = (value or "").strip()
    if value == "":
        return None
    return int(float(value))


def _row_to_label(row: dict[str, str]) -> LabelRecord:
    region = REGION_ALIASES.get((row.get("anatomical_region") or "").strip(), (row.get("anatomical_region") or "").strip())

    # Критерии: 1 = нарушен. Берём только колонки своей области, но если
    # заполнена чужая — тоже читаем (валидация при write это отвергнет).
    codes = []
    for column, code in VIOLATION_BY_CRITERION.items():
        value = (row.get(column) or "").strip()
        if value not in {"", "0"}:
            codes.append(code)

    return LabelRecord(
        path_to_study=(row.get("path_to_study") or "").strip(),
        study_uid=(row.get("study_uid") or "").strip(),
        image_uid=(row.get("image_uid") or "").strip(),
        anatomical_region=region,
        side=(row.get("side") or "none").strip() or "none",
        quality_class=_read_int_or_none(row.get("quality_class")),
        violation_type=";".join(codes) if codes else "none",
        notes=(row.get("notes") or "").strip(),
        exclude_from_training=(row.get("exclude_from_training") or "").strip().lower() == "true",
    )


def _label_to_row(label: LabelRecord) -> dict[str, str]:
    codes = set(violation_codes(label))
    labeled = label.quality_class is not None
    row = {column: "" for column in LABEL_COLUMNS}
    row["path_to_study"] = label.path_to_study
    row["study_uid"] = label.study_uid
    row["image_uid"] = label.image_uid
    row["anatomical_region"] = label.anatomical_region
    row["side"] = label.side
    row["quality_class"] = "" if label.quality_class is None else str(int(label.quality_class))
    for code, column in CRITERION_BY_VIOLATION.items():
        # неразмеченная строка хранит критерии пустыми
        row[column] = ("1" if code in codes else "0") if labeled else ""
    row["notes"] = label.notes
    row["exclude_from_training"] = "True" if label.exclude_from_training else "False"
    return row


def _sniff_delimiter(stream) -> str:
    """CSV приходит и с ';', и (исторически) с ',' — определяем по заголовку."""
    header = stream.readline()
    stream.seek(0)
    return ";" if header.count(";") > header.count(",") else ","


class LabelStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def read(self) -> list[LabelRecord]:
        with self._lock:
            if not self.path.exists():
                return []
            with self.path.open("r", encoding="utf-8-sig", newline="") as stream:
                delimiter = _sniff_delimiter(stream)
                result = [
                    _row_to_label(row)
                    for row in csv.DictReader(stream, delimiter=delimiter)
                    if any((value or "").strip() for value in row.values())
                ]
                return result

    def write(self, labels: list[LabelRecord]) -> None:
        for label in labels:
            validate_label(label)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=LABEL_COLUMNS, delimiter=LABEL_DELIMITER)
                writer.writeheader()
                writer.writerows(_label_to_row(item) for item in labels)
            os.replace(temporary, self.path)

    def update(self, image_uid: str, region: str, side: str, quality: int | None, violations: str,
               notes: str) -> LabelRecord:
        """Ручная правка одной строки разметки (область, сторона, класс, критерии)."""
        labels = self.read()
        for label in labels:
            if label.image_uid != image_uid:
                continue
            label.anatomical_region = REGION_ALIASES.get(region, region)
            label.side = side.strip()
            label.quality_class = quality if quality in {0, 1} else None
            label.violation_type = violations.strip()
            label.notes = notes.strip()
            validate_label(label)
            self.write(labels)
            return label
        raise KeyError(image_uid)
