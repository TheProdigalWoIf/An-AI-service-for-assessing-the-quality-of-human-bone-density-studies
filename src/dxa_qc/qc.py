from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from dxa_qc.dicom import normalized_pixels, read_dataset
from dxa_qc.models import QCResult


def principal_angle(mask: np.ndarray, weights: np.ndarray) -> float:
    y, x = np.nonzero(mask)
    if len(x) < 30:
        return 0.0
    w = weights[y, x] + 0.05
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    covariance = np.cov(np.vstack((x, y)), aweights=w)
    _, vectors = np.linalg.eigh(covariance)
    vx, vy = vectors[:, -1]
    return float(np.degrees(np.arctan2(vx, vy)))


def largest_component_ratio(mask: np.ndarray) -> float:
    sy = max(1, mask.shape[0] // 160)
    sx = max(1, mask.shape[1] // 160)
    reduced = mask[::sy, ::sx]
    seen = np.zeros_like(reduced, dtype=bool)
    largest = 0
    total = int(reduced.sum())
    for start_y, start_x in zip(*np.nonzero(reduced)):
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        size = 0
        while stack:
            y, x = stack.pop()
            size += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < reduced.shape[0] and 0 <= nx < reduced.shape[1]:
                    if reduced[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        largest = max(largest, size)
    return largest / max(total, 1)


def classify_region(image: np.ndarray) -> str:
    h, w = image.shape
    if float(np.std(image)) < 0.20 or float(np.mean(image < 0.08)) < 0.07:
        return "unknown"
    bone = image >= 0.55
    if largest_component_ratio(bone) < 0.36:
        return "unknown"
    y, x = np.nonzero(image > np.percentile(image, 78))
    if len(x) < 30:
        return "unknown"
    angle = abs(principal_angle(image > np.percentile(image, 78), image))
    central = float(np.mean((x > 0.30 * w) & (x < 0.70 * w)))
    vertical_span = float((np.percentile(y, 95) - np.percentile(y, 5)) / h)
    return "lumbar_spine" if angle < 18 and central > 0.54 and vertical_span > 0.62 else "proximal_femur"


def silhouette_width(mask: np.ndarray, start: float, end: float) -> float:
    h = mask.shape[0]
    widths = []
    for row in mask[int(start * h) : int(end * h)]:
        points = np.flatnonzero(row)
        if len(points) > 4:
            widths.append(points[-1] - points[0] + 1)
    return float(np.median(widths)) if widths else 0.0


def analyze_pixels(image: np.ndarray, path: str, study_uid: str, image_uid: str, started: float | None = None) -> QCResult:
    """Резервный правило-ориентированный анализ без модели.

    Сторону бедра правила НЕ определяют (нет экспертной разметки left/right):
    для бедра side='none' не выдумывается по координате, решение получает
    статус needs_review. Итоговое качество — по числу сработавших правил.
    """
    started = started or time.perf_counter()
    region = classify_region(image)
    if region == "unknown":
        return QCResult(
            path_to_study=path, study_uid=study_uid, image_uid=image_uid,
            anatomical_region=region, side="none",
            quality_class=1, violation_type="not_dxa_image",
            processing_status="Success",
            time_of_processing=round(time.perf_counter() - started, 4),
            quality_score=0.05,
            decision_status="needs_review",
        )

    h, w = image.shape
    bone = image >= 0.55
    y, x = np.nonzero(bone)
    angle = principal_angle(bone, image)
    vertical_span = float((np.percentile(y, 98) - np.percentile(y, 2)) / h)
    horizontal_span = float((np.percentile(x, 98) - np.percentile(x, 2)) / w)
    border = max(3, int(min(h, w) * 0.025))
    border_mask = np.zeros_like(bone)
    border_mask[:border] = border_mask[-border:] = True
    border_mask[:, :border] = border_mask[:, -border:] = True
    border_contact = float(np.sum(bone & border_mask) / max(np.sum(bone), 1))
    violations: list[str] = []

    if region == "lumbar_spine":
        top = float(np.mean(bone[: int(0.22 * h), int(0.30 * w) : int(0.70 * w)]))
        left = float(np.mean(bone[int(0.72 * h) :, : int(0.38 * w)]))
        right = float(np.mean(bone[int(0.72 * h) :, int(0.62 * w) :]))
        if top < 0.12:
            violations.append("spine_th12_not_visualized")
        if min(left, right) < 0.004:
            violations.append("iliac_crests_not_visualized")
        if abs(angle) > 5:
            violations.append("spine_axis_misalignment")
    else:
        if vertical_span < 0.82 or horizontal_span < 0.38:
            violations.append("hip_anatomy_incomplete")
        if border_contact > 0.115:
            violations.append("hip_roi_margin_insufficient")
        ratio = silhouette_width(bone, 0.40, 0.68) / max(silhouette_width(bone, 0.72, 0.92), 1.0)
        if ratio < 0.55:
            violations.append("hip_overrotation_suspected")
        elif ratio > 1.80:
            violations.append("hip_underrotation_suspected")

    if float(np.mean(image > 0.98)) > 0.06:
        violations.append("foreign_object_suspected")
    if float(np.percentile(image, 95) - np.percentile(image, 10)) < 0.40:
        violations.append("low_contrast")
    score = float(np.clip(1.0 - 0.18 * len(violations) - min(border_contact, 0.18), 0.02, 0.99))
    return QCResult(
        path_to_study=path, study_uid=study_uid, image_uid=image_uid,
        anatomical_region=region,
        side="none",  # правила не знают стороны; бедро помечается needs_review ниже
        quality_class=int(bool(violations)),
        violation_type=";".join(violations) if violations else "none",
        processing_status="Success",
        time_of_processing=round(time.perf_counter() - started, 4),
        quality_score=round(score, 3),
        angle_degrees=round(angle, 2) if region == "lumbar_spine" else None,
        decision_status="final" if region == "lumbar_spine" else "needs_review",
    )


def analyze_file(path: Path, root: Path) -> QCResult:
    started = time.perf_counter()
    relative = str(path.relative_to(root))
    try:
        dataset = read_dataset(path)
        return analyze_pixels(
            normalized_pixels(dataset), relative,
            str(getattr(dataset, "StudyInstanceUID", path.parent.name)),
            str(getattr(dataset, "SOPInstanceUID", path.stem)), started,
        )
    except Exception as exc:
        return QCResult(
            path_to_study=relative, study_uid=path.parent.name, image_uid=path.stem,
            anatomical_region="unknown", side="none",
            quality_class=1, violation_type=f"read_error:{type(exc).__name__}",
            processing_status="Failure",
            time_of_processing=round(time.perf_counter() - started, 4),
            quality_score=0.0,
            decision_status="needs_review",
        )
