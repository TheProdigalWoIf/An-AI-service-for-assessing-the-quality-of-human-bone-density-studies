"""Инференс модели контроля качества DXA-снимков (точка входа для API).

Логика повторяет схему экспертизы: снимок -> анатомическая область
(lumbar_spine/proximal_femur) -> сторона (none/right/left, только для бедра)
-> пять критериев -> ИТОГОВОЕ РЕШЕНИЕ.

Критерий «ось позвоночника» — детерминированный геометрический: наклон
средней линии самой колонки позвоночника (норма по методичке ≤5°). Ось
рисуется по колонке, а не по всему кадру — как в экспертной разметке, поэтому
сколиоз с вертикальной осью и наклонённый прямой позвоночник различаются.
Сеть этот критерий больше не назначает — на 9 размечанных примерах голова
оси не выучилась (AUC ~0.5). Остальные критерии решает сеть.

Плохой снимок всегда получает конкретную причину: если ни один критерий не
превысил порог, берётся самый вероятный применимый критерий, а решение
получает статус needs_review. Расхождение геометрии и сети по оси также
уводит решение в needs_review. Если чекпойнта нет — прозрачный откат на
резервные правила qc.py.
"""
from __future__ import annotations

import time
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dxa_qc.config import settings
from dxa_qc.dicom import normalize_array, normalized_pixels, read_dataset
from dxa_qc.geometry import spine_axis_metrics
from dxa_qc.ml import (
    REGIONS,
    SIDES,
    VIOLATIONS,
    allowed_violations,
    choose_device,
    image_transform,
    load_checkpoint,
)
from dxa_qc.models import QCResult
from dxa_qc.qc import analyze_file as analyze_with_rules

# Норма отклонения оси позвоночника по методичке (градусы).
SPINE_AXIS_LIMIT_DEG = 5.0


@lru_cache(maxsize=1)  # первый вызов грузит модель, остальные получают готовый кортеж
def model_bundle():
    device = choose_device()
    model, checkpoint = load_checkpoint(settings.model_path, device)
    return model, checkpoint, device, image_transform(False)  # False = без аугментаций


def model_available() -> bool:
    return settings.model_path.is_file()


def predict_pixels(pixels: np.ndarray) -> tuple[str, str, int, str, float, str, float | None]:
    """(область, сторона, качество, причины, уверенность, статус решения, угол оси)."""
    model, checkpoint, device, transform = model_bundle()

    tensor = transform(pixels).unsqueeze(0).to(device)

    with torch.inference_mode():
        region_logits, side_logits, quality_logit, violation_logits = model(tensor)

    region_probability = torch.softmax(region_logits, dim=1)[0].cpu().numpy()
    side_probability = torch.softmax(side_logits, dim=1)[0].cpu().numpy()
    quality_probability = float(torch.sigmoid(quality_logit)[0].cpu())
    violation_probability = torch.sigmoid(violation_logits)[0].cpu().numpy()

    region = REGIONS[int(np.argmax(region_probability))]

    # Сторона — самостоятельная голова (экспертная разметка left/right),
    # для позвоночника сторона всегда none.
    if region == "lumbar_spine":
        side = "none"
    else:
        side = SIDES[int(np.argmax(side_probability))]

    quality_threshold = float(checkpoint["quality_threshold"])
    network_bad = quality_probability >= quality_threshold

    # Ось позвоночника — детерминированный геометрический критерий: наклон
    # средней линии самой колонки (конвенция экспертной разметки: ось рисуется
    # по колонке, а не по всему кадру; норма по методичке ≤5°). Сеть ось не
    # решает — на 9 примерах её голова оси не выучилась (AUC ~0.5).
    metrics = spine_axis_metrics(pixels) if region == "lumbar_spine" else None
    angle = metrics["tilt_deg"] if metrics else None
    axis_violated = angle is not None and abs(angle) > SPINE_AXIS_LIMIT_DEG

    quality = int(network_bad or axis_violated)  # 1 = плохой
    certainty = quality_probability if network_bad else 1.0 - quality_probability

    selected: list[str] = []
    decision_status = "final"

    if certainty < 0.60:
        # Вероятность решения близка к порогу — честно помечаем как
        # требующее проверки, а не уверенный вердикт.
        decision_status = "needs_review"

    if quality:
        allowed = allowed_violations(region)
        thresholds = checkpoint["violation_thresholds"]
        # Ось назначает только геометрия — из сетевых кандидатов её исключаем.
        candidates = [
            index for index, name in enumerate(VIOLATIONS)
            if name in allowed and name != "spine_axis_misalignment"
        ]
        selected = [
            VIOLATIONS[index]
            for index in candidates
            if violation_probability[index] >= float(thresholds[index])
        ]
        if axis_violated:
            selected.append("spine_axis_misalignment")
            if not network_bad:
                # Геометрия говорит «ось нарушена», сеть — «хороший»: расхождение
                # источников — решение требует проверки человеком.
                decision_status = "needs_review"
        if not selected:
            # Плохой снимок обязан иметь причину: берём самый вероятный
            # применимый критерий, но решение помечаем как требующее проверки.
            selected = [VIOLATIONS[max(candidates, key=lambda i: violation_probability[i])]]
            decision_status = "needs_review"

    return (
        region,
        side,
        quality,
        ";".join(selected) if selected else "none",
        certainty,
        decision_status,
        round(angle, 2) if angle is not None else None,
    )


def analyze_uploaded_image(data: bytes, filename: str) -> QCResult:
    """Анализ файла, загруженного из браузера в формате JPEG/PNG/WebP."""
    started = time.perf_counter()
    image_uid = sha256(data).hexdigest()[:24]

    try:
        with Image.open(BytesIO(data)) as image:
            pixels = normalize_array(np.asarray(image.convert("RGB")))
        region, side, quality, violations, certainty, decision_status, angle = predict_pixels(pixels)

        return QCResult(
            path_to_study=filename,
            study_uid="browser_upload",
            image_uid=image_uid,
            anatomical_region=region,
            side=side,
            quality_class=quality,
            violation_type=violations,
            processing_status="Success",
            time_of_processing=round(time.perf_counter() - started, 4),
            quality_score=round(certainty, 3),
            angle_degrees=angle,
            model_source="resnet18_multitask",
            decision_status=decision_status,
        )

    except Exception as exc:
        return QCResult(
            path_to_study=filename,
            study_uid="browser_upload",
            image_uid=image_uid,
            anatomical_region="unknown",
            side="none",
            quality_class=1,
            violation_type=f"read_error:{type(exc).__name__}",
            processing_status="Failure",
            time_of_processing=round(time.perf_counter() - started, 4),
            quality_score=0.0,
            model_source="resnet18_multitask",
            decision_status="needs_review",
        )


def analyze_file(path: Path, root: Path) -> QCResult:
    """Анализ DICOM-файла из каталога данных; без модели — правила qc.py."""
    if not model_available():
        return analyze_with_rules(path, root)

    started = time.perf_counter()
    relative = str(path.relative_to(root))

    try:
        dataset = read_dataset(path)
        region, side, quality, violations, certainty, decision_status, angle = predict_pixels(
            normalized_pixels(dataset)
        )

        return QCResult(
            path_to_study=relative,
            study_uid=str(getattr(dataset, "StudyInstanceUID", path.parent.name)),
            image_uid=str(getattr(dataset, "SOPInstanceUID", path.stem)),
            anatomical_region=region,
            side=side,
            quality_class=quality,
            violation_type=violations,
            processing_status="Success",
            time_of_processing=round(time.perf_counter() - started, 4),
            quality_score=round(certainty, 3),
            angle_degrees=angle,
            model_source="resnet18_multitask",
            decision_status=decision_status,
        )

    except Exception as exc:
        return QCResult(
            path_to_study=relative,
            study_uid=path.parent.name,
            image_uid=path.stem,
            anatomical_region="unknown",
            side="none",
            quality_class=1,
            violation_type=f"read_error:{type(exc).__name__}",
            processing_status="Failure",
            time_of_processing=round(time.perf_counter() - started, 4),
            quality_score=0.0,
            model_source="resnet18_multitask",
            decision_status="needs_review",
        )
