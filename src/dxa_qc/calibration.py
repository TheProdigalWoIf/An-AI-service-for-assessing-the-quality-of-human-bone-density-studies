# -*- coding: utf-8 -*-
"""Инференс калибровочной модели контроля качества DXA-снимков.

Модель обучена скриптом scripts/train_calibration.py на калибровочной разметке
(разметка.xlsx, лист «Калибровка») по правилам «Методических рекомендаций по
проведению ДРА» (НПКЦ ДиТ ДЗМ, 2022). Этот модуль — общая точка использования
обученной модели: и веб-приложением (dxa_qc.inference), и скриптом обучения.

Ключевые особенности:
* четыре головы: часть тела (позвоночник/правое бедро/левое бедро), класс
  качества, 7 технических нарушений, отдельная «клиническая» голова;
* клинические отклонения (сколиоз, перелом, эндопротез, люмбализация)
  НЕ считаются нарушением качества снимка и сообщаются отдельно;
* угол отклонения оси позвоночника считается по определению из задания:
  угол между перпендикуляром (горизонталью), опущенным к остистому отростку
  нижнего поясничного позвонка, и прямой, соединяющей остистые отростки
  первого и последнего позвонков; критерий годности из таблицы — до 5°.
"""
from __future__ import annotations

import math
import time
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

from dxa_qc.config import settings
from dxa_qc.dicom import normalize_array, normalized_pixels, read_dataset
from dxa_qc.models import QCResult

# ----------------------------------------------------------------------------
# Словари модели
# ----------------------------------------------------------------------------

# Имя DICOM-файла -> анатомическая часть тела (соглашение калибровочных данных).
PART_BY_FILENAME = {"п.dcm": "spine", "пб.dcm": "hip_right", "лб.dcm": "hip_left"}
PARTS = ("spine", "hip_right", "hip_left")
PART_TITLES = {"spine": "Позвоночник", "hip_right": "Правое бедро", "hip_left": "Левое бедро"}
# Код части -> код анатомической области, понятный остальным частям приложения
# (geometry.py строит разметку позвоночника для lumbar_spine, для остальных — бедро).
REGION_BY_PART = {
    "spine": "lumbar_spine",
    "hip_right": "proximal_femur_right",
    "hip_left": "proximal_femur_left",
}

# Таксономия технических нарушений (совпадает с scripts/train_calibration.py).
VIOLATIONS = (
    "spine_positioning",       # позвоночник: некорректная укладка
    "spine_axis_over_5deg",    # позвоночник: ось выровнена хуже 5°
    "spine_foreign_objects",   # позвоночник: посторонние предметы/артефакты/наложения
    "hip_right_rotation",      # правое бедро: позиционирование/ротация
    "hip_right_roi",           # правое бедро: некорректная область интереса
    "hip_left_rotation",       # левое бедро: позиционирование/ротация
    "hip_left_roi",            # левое бедро: некорректная область интереса
)
# Какие метки нарушений применимы к каждой части тела.
PART_VIOLATION_MASK = {
    "spine": [0, 1, 2],
    "hip_right": [3, 4],
    "hip_left": [5, 6],
}

# Пороги решения (как при обучении/валидации: качество и клиника — 0.5).
QUALITY_THRESHOLD = 0.5
CLINICAL_THRESHOLD = 0.5
VIOLATION_THRESHOLD = 0.5
AXIS_LIMIT_DEG = 5.0  # критерий «ось выровнена» из таблицы разметки: до 5°
MODEL_SOURCE = "calibration_v1"  # метка источника анализа в QCResult

# Русские названия нарушений для интерфейса и отчётов.
VIOLATION_TITLES = {
    "spine_positioning": "некорректная укладка позвоночника",
    "spine_axis_over_5deg": "ось позвоночника отклонена более чем на 5°",
    "spine_foreign_objects": "посторонние предметы, артефакты или наложения",
    "hip_right_rotation": "ротация/позиционирование правого бедра",
    "hip_right_roi": "область интереса правого бедра некорректна",
    "hip_left_rotation": "ротация/позиционирование левого бедра",
    "hip_left_roi": "область интереса левого бедра некорректна",
}


# ----------------------------------------------------------------------------
# Архитектура (должна совпадать с обучением — см. train_calibration.py)
# ----------------------------------------------------------------------------

class CalibrationModel(torch.nn.Module):
    """ResNet18 + четыре головы: часть тела, качество, нарушения, клиника.

    Магистраль может инициализироваться весами доменной модели проекта
    (models/dxa_qc_resnet18.pt) — так делалось при обучении; при инференсе
    веса целиком приходят из чекпойнта calibration_resnet18.pt.
    """

    def __init__(self, domain_checkpoint: Path | None = None):
        super().__init__()
        backbone_state = None
        if domain_checkpoint is not None and domain_checkpoint.is_file():
            backbone_state = {k[len("backbone."):]: v for k, v in
                              torch.load(domain_checkpoint, map_location="cpu",
                                         weights_only=False)["model_state"].items()
                              if k.startswith("backbone.")}
        backbone = models.resnet18(
            weights=None if backbone_state is not None else models.ResNet18_Weights.IMAGENET1K_V1)
        features = backbone.fc.in_features
        # Сначала отделяем классификатор, затем загружаем веса: в доменном
        # чекпойнте fc уже заменён на Identity, strict-загрузка в «сырой» ResNet
        # падает из-за отсутствующих ключей fc.weight/fc.bias.
        backbone.fc = torch.nn.Identity()
        if backbone_state is not None:
            backbone.load_state_dict(backbone_state)
        self.backbone = backbone
        self.part_head = torch.nn.Linear(features, len(PARTS))
        self.quality_head = torch.nn.Linear(features, 1)
        self.violation_head = torch.nn.Linear(features, len(VIOLATIONS))
        self.clinical_head = torch.nn.Linear(features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return (self.part_head(features),
                self.quality_head(features).squeeze(1),
                self.violation_head(features),
                self.clinical_head(features).squeeze(1))


# ----------------------------------------------------------------------------
# Угол отклонения оси позвоночника (по определению из задания)
# ----------------------------------------------------------------------------

def spine_axis_angle(image: np.ndarray) -> tuple[float, float, dict]:
    """Угол между перпендикуляром к оси сканирования, опущенным к остистому
    отростку нижнего поясничного позвонка, и прямой, соединяющей остистые
    отростки первого и последнего позвонков.

    Положение остистых отростков по строкам кадра оценивается как срединная
    линия позвоночного столба (взвешенный центроид ярких пикселей центральной
    полосы); первый/последний позвонки — 20-я/80-я перцентили вертикального
    охвата столба. Это приближение, а не экспертная разметка.

    Возвращает (построенный_угол, отклонение_от_вертикали, детали):
    у идеально прямой оси прямая отростков вертикальна, построенный угол с
    горизонталью равен 90°, отклонение = |угол − 90°| = 0.
    """
    h, w = image.shape
    x0, x1 = int(w * 0.30), int(w * 0.70)          # центральная полоса кадра
    band = image[:, x0:x1]
    threshold = float(np.percentile(band, 75))      # уровень «кости»

    midline: dict[int, float] = {}
    for y in range(h):
        row = band[y]
        mask = row >= threshold
        if int(mask.sum()) >= 4:                    # строка с достаточной опорой
            idx = np.flatnonzero(mask) + x0
            weights = row[mask] + 0.05
            midline[y] = float(np.average(idx, weights=weights))

    if len(midline) < h // 4:
        return float("nan"), float("nan"), {"reason": "позвоночный столб не найден"}

    ys = np.array(sorted(midline))
    xs = np.array([midline[y] for y in ys])
    window = max(3, h // 60) | 1                    # скользящая медиана — шум строк
    xs_smooth = np.array([np.median(xs[max(0, i - window // 2): i + window // 2 + 1])
                          for i in range(len(xs))])

    def point_at(fraction: float) -> tuple[float, float]:
        y_target = float(np.percentile(ys, fraction * 100))
        i = int(np.argmin(np.abs(ys - y_target)))
        half = max(2, len(ys) // 50)
        lo, hi = max(0, i - half), min(len(ys), i + half + 1)
        return float(np.mean(xs_smooth[lo:hi])), float(np.mean(ys[lo:hi]))

    x_first, y_first = point_at(0.20)   # остистый отросток первого позвонка (L1)
    x_last, y_last = point_at(0.80)     # остистый отросток последнего (L5)

    dx, dy = x_last - x_first, y_last - y_first
    line_angle = float(np.degrees(np.arctan2(abs(dy), abs(dx))))  # угол с горизонталью
    deviation = abs(line_angle - 90.0)  # отклонение оси от вертикали
    details = {"p_first": (round(x_first, 1), round(y_first, 1)),
               "p_last": (round(x_last, 1), round(y_last, 1)),
               "angle_with_horizontal": round(line_angle, 2)}
    return line_angle, deviation, details


# ----------------------------------------------------------------------------
# Загрузка модели и предсказание
# ----------------------------------------------------------------------------

def _transform() -> transforms.Compose:
    """Тот же пайплайн предобработки, что при обучении (без аугментаций)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


@lru_cache(maxsize=1)
def model_bundle():
    """Ленивая загрузка модели ровно один раз на процесс."""
    checkpoint = torch.load(settings.calibration_path, map_location="cpu", weights_only=False)
    model = CalibrationModel()
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, _transform()


def model_available() -> bool:
    return settings.calibration_path.is_file()


def predict_pixels(pixels: np.ndarray) -> dict:
    """Полное предсказание для нормализованного изображения [0..1]."""
    model, transform = model_bundle()
    tensor = transform(pixels).unsqueeze(0)
    with torch.inference_mode():
        part_logits, quality_logit, violation_logits, clinical_logit = model(tensor)
    part_prob = torch.softmax(part_logits, 1)[0].numpy()
    part = PARTS[int(np.argmax(part_prob))]
    p_quality = float(torch.sigmoid(quality_logit)[0])
    violation_prob = torch.sigmoid(violation_logits)[0].numpy()
    p_clinical = float(torch.sigmoid(clinical_logit)[0])

    quality = int(p_quality >= QUALITY_THRESHOLD)
    applicable = PART_VIOLATION_MASK[part]
    selected = [VIOLATIONS[i] for i in applicable if violation_prob[i] >= VIOLATION_THRESHOLD]
    if quality and not selected:
        # Снимок некачественный, но ни одна метка не превысила порог — берём
        # наиболее вероятное применимое нарушение (для оператора важен код).
        selected = [max((violation_prob[i], VIOLATIONS[i]) for i in applicable)[1]]

    result = {
        "part": part,
        "region": REGION_BY_PART[part],
        "part_title": PART_TITLES[part],
        "quality": quality,
        "p_quality": p_quality,
        "violations": selected,
        "p_clinical": p_clinical,
        "clinical_flag": p_clinical >= CLINICAL_THRESHOLD,
        "certainty": p_quality if quality else 1.0 - p_quality,
    }
    if part == "spine":
        # Угол считается только для позвоночника (определение про поясничные позвонки).
        constructed, deviation, details = spine_axis_angle(pixels)
        result.update({
            "axis_angle_deg": None if math.isnan(constructed) else round(constructed, 2),
            "axis_deviation_deg": None if math.isnan(deviation) else round(deviation, 2),
            "axis_ok": None if math.isnan(deviation) else bool(deviation <= AXIS_LIMIT_DEG),
            "axis_details": details,
        })
    return result


def _to_qc_result(prediction: dict, started: float, path: str, study_uid: str,
                  image_uid: str) -> QCResult:
    """Сборка QCResult с полями калибровочной модели (клиника — отдельно)."""
    return QCResult(
        path, study_uid, image_uid,
        prediction["region"],
        prediction["quality"],
        ";".join(prediction["violations"]) if prediction["violations"] else "none",
        "Success",
        round(time.perf_counter() - started, 4),
        round(prediction["certainty"], 3),
        prediction.get("axis_angle_deg"),
        MODEL_SOURCE,
        # Клиническое отклонение: если голова сработала — код-признак; текстовое
        # описание известно только из таблицы разметки, здесь — вероятность.
        "suspected_clinical_finding" if prediction["clinical_flag"] else "",
        round(prediction["p_clinical"], 3),
        prediction.get("axis_deviation_deg"),
    )


def analyze_file(path: Path, root: Path) -> QCResult:
    """Анализ DICOM-файла из каталога данных калибровочной моделью."""
    started = time.perf_counter()
    relative = str(path.relative_to(root))
    try:
        dataset = read_dataset(path)
        prediction = predict_pixels(normalized_pixels(dataset))
        return _to_qc_result(prediction, started, relative,
                             str(getattr(dataset, "StudyInstanceUID", path.parent.name)),
                             str(getattr(dataset, "SOPInstanceUID", path.stem)))
    except Exception as exc:
        return QCResult(relative, path.parent.name, path.stem, "unknown", 1,
                        f"read_error:{type(exc).__name__}", "Failure",
                        round(time.perf_counter() - started, 4), 0.0,
                        None, MODEL_SOURCE)


def analyze_uploaded_image(data: bytes, filename: str) -> QCResult:
    """Анализ файла, загруженного из браузера (DICOM обрабатывает api.py во
    временном файле через analyze_file; здесь — JPEG/PNG/WebP через Pillow)."""
    started = time.perf_counter()
    from hashlib import sha256
    image_uid = sha256(data).hexdigest()[:24]
    try:
        with Image.open(BytesIO(data)) as image:
            pixels = normalize_array(np.asarray(image.convert("RGB")))
        prediction = predict_pixels(pixels)
        return _to_qc_result(prediction, started, filename, "browser_upload", image_uid)
    except Exception as exc:
        return QCResult(filename, "browser_upload", image_uid, "unknown", 1,
                        f"read_error:{type(exc).__name__}", "Failure",
                        round(time.perf_counter() - started, 4), 0.0,
                        None, MODEL_SOURCE)
