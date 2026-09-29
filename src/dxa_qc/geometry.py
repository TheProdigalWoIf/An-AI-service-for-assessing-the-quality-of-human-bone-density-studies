from __future__ import annotations

import math

import numpy as np


def _weighted_axis(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape
    threshold = float(np.percentile(image, 78))
    weights = np.square(np.clip(image - threshold, 0.0, None))
    # Suppress borders and favor the anatomical center of DXA frames.
    mask = np.zeros_like(weights)
    mask[int(height * 0.04):int(height * 0.96), int(width * 0.08):int(width * 0.92)] = 1.0
    weights *= mask
    if float(weights.sum()) <= 1e-8:
        weights = np.square(np.clip(image, 0.0, None)) * mask
    y, x = np.indices(image.shape)
    total = float(weights.sum())
    if total <= 1e-8:
        center = np.array([width / 2.0, height / 2.0])
        return center, np.array([0.0, 1.0]), weights
    center = np.array([(x * weights).sum() / total, (y * weights).sum() / total])
    dx, dy = x - center[0], y - center[1]
    covariance = np.array([
        [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
        [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
    ]) / total
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    if direction[1] < 0:
        direction = -direction
    return center, direction, weights


def _point(center: np.ndarray, direction: np.ndarray, distance: float) -> list[float]:
    value = center + direction * distance
    return [round(float(value[0]), 2), round(float(value[1]), 2)]


def axis_angle(image: np.ndarray) -> float:
    """Угол взвешенной главной оси всего кадра (градусы)."""
    _, direction, _ = _weighted_axis(image)
    return math.degrees(math.atan2(float(direction[0]), float(direction[1])))


def spine_midline(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Средняя линия колонки позвоночника: центроид ярких пикселей центральной
    полосы по строкам (конвенция эксперта: ось рисуется по самой колонке,
    а не по всему кадру)."""
    height, width = image.shape
    threshold = float(np.percentile(image, 78))
    x0, x1 = int(0.28 * width), int(0.72 * width)
    ys, xs = [], []
    for y in range(int(0.10 * height), int(0.90 * height)):
        band = image[y, x0:x1]
        mask = band > threshold
        if mask.sum() < max(10, (x1 - x0) // 20):
            continue
        ys.append(float(y))
        xs.append(float(np.nonzero(mask)[0].mean()) + x0)
    return np.asarray(ys), np.asarray(xs)


def _line_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Наклон dx/dy (0 = вертикаль); при малом числе точек — вертикаль."""
    if len(y) < 8:
        return 0.0
    return float(np.polyfit(y, x, 1)[0])


def spine_axis_metrics(image: np.ndarray) -> dict:
    """Геометрия колонки: наклон, угол изгиба (верх/низ), боковое отклонение.

    Наклон — угол прямой, аппроксимирующей среднюю линию колонки: именно он
    используется как критерий оси (норма ≤5°). Угол изгиба и отклонение —
    прокси сколиоза (угол между верхним и нижним участками колонки, как в
    экспертной разметке двумя линиями).
    """
    y, x = spine_midline(image)
    if len(y) < 20:
        return {"tilt_deg": 0.0, "segment_angle_deg": 0.0, "deviation": 0.0, "points": int(len(y))}
    tilt = math.degrees(math.atan(_line_slope(y, x)))
    span = float(y.max() - y.min())
    upper = y <= y.min() + 0.40 * span
    lower = y >= y.max() - 0.40 * span
    segment_angle = math.degrees(abs(
        math.atan(_line_slope(y[upper], x[upper])) - math.atan(_line_slope(y[lower], x[lower]))
    ))
    chord = x[0] + (x[-1] - x[0]) * (y - y[0]) / max(span, 1.0)
    deviation = float(np.max(np.abs(x - chord)) / image.shape[1])
    return {"tilt_deg": tilt, "segment_angle_deg": segment_angle, "deviation": deviation,
            "points": int(len(y))}


def build_qc_annotation(image: np.ndarray, region: str) -> dict:
    height, width = image.shape

    if region == "lumbar_spine":
        # Ось рисуется по средней линии самой колонки (конвенция экспертной
        # разметки), угол — наклон аппроксимирующей прямой.
        ys, xs = spine_midline(image)
        metrics = spine_axis_metrics(image)
        if len(ys) >= 2:
            axis = [
                [round(float(xs[0]), 2), round(float(ys[0]), 2)],
                [round(float(xs[-1]), 2), round(float(ys[-1]), 2)],
            ]
            indices = np.linspace(0, len(ys) - 1, 24).astype(int)
            midline = [[round(float(xs[i]), 2), round(float(ys[i]), 2)] for i in indices]
        else:
            axis = [[width / 2.0, 0.0], [width / 2.0, float(height)]]
            midline = axis
        result = {
            "kind": "lumbar_geometry",
            "width": int(width),
            "height": int(height),
            "angle_degrees": round(metrics["tilt_deg"], 2),
            "segment_angle_degrees": round(metrics["segment_angle_deg"], 2),
            "axis": axis,
            "midline": midline,
            "guides": [],
            "roi": None,
            "method": "spine_column_midline",
        }
        for index, fraction in enumerate((0.22, 0.40, 0.58, 0.76), start=1):
            y = height * fraction
            if len(ys) >= 2 and ys.min() <= y <= ys.max():
                guide_center = float(np.interp(y, ys, xs))
            else:
                guide_center = width / 2.0
            start = [round(guide_center - width * 0.38, 2), round(y, 2)]
            end = [round(guide_center + width * 0.38, 2), round(y, 2)]
            result["guides"].append({
                "start": start, "end": end, "label": f"Уровень {index}",
                "length_px": round(math.dist(start, end), 1),
            })
        return result

    center, direction, weights = _weighted_axis(image)
    angle = math.degrees(math.atan2(float(direction[0]), float(direction[1])))
    half = min(height * 0.44, width * 1.1)
    axis = [_point(center, direction, -half), _point(center, direction, half)]
    result = {
        "kind": "hip_geometry",
        "width": int(width),
        "height": int(height),
        "angle_degrees": round(angle, 2),
        "axis": axis,
        "guides": [],
        "roi": None,
        "method": "intensity_weighted_principal_axis",
    }
    positive = np.argwhere(weights > 0)
    if positive.size:
        y0, x0 = np.percentile(positive, 5, axis=0)
        y1, x1 = np.percentile(positive, 95, axis=0)
        result["roi"] = {
            "x": round(float(x0), 2), "y": round(float(y0), 2),
            "width": round(float(x1 - x0), 2), "height": round(float(y1 - y0), 2),
        }
    return result
