#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Обучение модели контроля качества DXA-снимков на калибровочной разметке.

Данные
------
* Каталог ``Исследования/<№>/...`` — DICOM-снимки; имя файла задаёт часть тела:
  ``п.dcm`` — поясничный отдел позвоночника (AP Spine),
  ``пб.dcm`` — проксимальный отдел правого бедра, ``лб.dcm`` — левого.
* Таблица ``разметка.xlsx`` (лист «Калибровка») — эталонные ответы разметчика:
  технические флаги по каждой части тела, итоговый класс качества и комментарий.
  Сопоставление таблицы и файлов выполняется по номеру исследования (№ = имя папки):
  StudyInstanceUID в файлах относится к другому стандарту (1.2.643...), чем
  идентификаторы в таблице (2.25...), поэтому прямой матчинг по UID невозможен.

Правила формирования правильных ответов (y без циркумфлекса)
------------------------------------------------------------
1. y_true для расчёта loss берётся из столбцов «Итог» таблицы — это эталонный
   класс качества снимка (0 — годен, 1 — нарушение качества).
2. «Клинические отклонения не являются нарушениями снимка» (сколиоз, перелом,
   эндопротез, люмбализация). В таблице есть три строки, где «Итог» расходится
   с техническими флагами именно из-за клиники:
     №6, №11 — итог=1 при всех технических флагах=0 и комментарии «сколиоз»
     №35     — итог=0 при флаге «ось»=1 и комментарии «перелом»
   Для №6 и №11 итоговый класс скорректирован до 0: единственной причиной
   «нарушения» была клиника. Флаги нарушений (укладка/ось/посторонние,
   ротация/ROI) остаются эталонами для соответствующих голов модели.
3. Клинические отклонения из комментариев выносятся в ОТДЕЛЬНУЮ колонку
   выходного отчёта и в лосс качества не входят; для их распознавания обучается
   отдельная бинарная голова «клиническое отклонение».

Критерии качества — по «Методическим рекомендациям по проведению
двухэнергетической рентгеновской абсорбциометрии» (НПКЦ ДиТ ДЗМ, 2022):
* позвоночник (п. 2.6): позвоночник по центру кадра, видна ~половина Th12,
  L4 полностью; ось выровнена (отклонение до 5°); без посторонних предметов
  и артефактов (имплантаты и т.п.);
* бедро (п. 2.7 и приложение Д): ≥3 см тканей над большим вертелом и под
  седалищной костью, достаточная внутренняя ротация (малый вертел минимально
  выступает), корректная область интереса (ROI шейки не захватывает вертел).

Угол отклонения оси позвоночника
---------------------------------
Считается по определению из задания: угол между
  (а) перпендикуляром, опущенным к остистому отростку нижнего поясничного
      позвонка (горизонталь, перпендикулярная вертикальной оси сканирования), и
  (б) прямой, соединяющей остистые отростки первого и последнего позвонков.
Положение остистых отростков по строкам кадра оценивается как срединная линия
позвоночного столба (взвешенный центроид ярких пикселей в центральной полосе);
первый/последний позвонки — по 20-й/80-й перцентилям вертикального охвата
столба (приближение, не экспертная разметка). Отклонение оси от вертикали =
|угол − 90°|; критерий годности из таблицы — не более 5°.

Запуск
------
    PYTHONPATH=src .venv/Scripts/python.exe scripts/train_calibration.py
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import openpyxl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Архитектура модели, угол по остистым отросткам и словари таксономии живут в
# пакете — это единый источник для обучения и веб-инференса (dxa_qc.calibration).
from dxa_qc.calibration import (  # noqa: E402
    AXIS_LIMIT_DEG,
    PART_BY_FILENAME,
    PART_TITLES,
    PART_VIOLATION_MASK,
    VIOLATIONS,
    CalibrationModel,
    spine_axis_angle,
)
from dxa_qc.dicom import normalized_pixels, read_dataset  # переиспользуем чтение DICOM

warnings.filterwarnings("ignore")  # pydicom ругается на длинные UID поставщика

# ----------------------------------------------------------------------------
# 1. Константы, специфичные для обучения
# ----------------------------------------------------------------------------

# Словари таксономии (VIOLATIONS, PART_BY_FILENAME, PART_VIOLATION_MASK,
# PART_TITLES), модель CalibrationModel и угол оси spine_axis_angle импортируются
# из dxa_qc.calibration — см. импорт вверху; здесь только обученческие правила.

# Ключевые слова комментариев, означающие КЛИНИЧЕСКОЕ отклонение (не нарушение
# качества снимка) — выносятся в отдельную колонку отчёта.
CLINICAL_KEYWORDS = ("сколиоз", "люмбализация", "l6", "перелом", "эндопротез")


def classify_comment(comment: str) -> tuple[bool, str]:
    """Комментарий таблицы -> (клиническое?, нормализованный текст)."""
    if not comment:
        return False, ""
    text = str(comment).strip().lower()
    return any(k in text for k in CLINICAL_KEYWORDS), str(comment).strip()


def clinical_targets(part: str, is_clinical: bool, comment: str, study_parts: set) -> tuple[bool, str]:
    """Привязка клинического комментария к конкретному снимку исследования.

    Правило: если в комментарии названа часть тела («правое бедро...») —
    комментарий относится к её снимку; если такого снимка в исследовании нет,
    клиника видна на снимке позвоночника (например, эндопротез правого бедра
    в кадре AP Spine — в таблице это флаг «посторонние предметы»); комментарии
    без части тела (сколиоз, люмбализация, перелом, «эндопротез ТБС»)
    относятся к снимку позвоночника.

    Возвращает (флаг для обучения клинической головы, текст для отдельной
    колонки отчёта). Текст возвращается ТОЛЬКО для снимка, к которому
    привязано отклонение, — чтобы не размазывать клинику по чужим снимкам.
    """
    if not is_clinical:
        return False, ""
    low = comment.lower()
    mentioned = None
    if "прав" in low and "бедро" in low:
        mentioned = "hip_right"
    elif "лев" in low and "бедро" in low:
        mentioned = "hip_left"
    target = mentioned if mentioned in study_parts else "spine"
    return part == target, comment if part == target else ""


# ----------------------------------------------------------------------------
# 2. Чтение таблицы разметки и сборка поизображенческого датасета
# ----------------------------------------------------------------------------

def read_labels(xlsx_path: Path) -> dict[int, dict]:
    """Лист «Калибровка» -> {номер исследования: метки}."""
    ws = openpyxl.load_workbook(xlsx_path, data_only=True)["Калибровка"]
    labels: dict[int, dict] = {}
    for row in ws.iter_rows(min_row=3, values_only=True):  # строки 1-2 — шапка
        num, study = row[0], row[1]
        if num is None or study is None:
            continue
        labels[int(num)] = {
            # Технические флаги: 1 = нарушение критерия; None = снимка такой
            # части в исследовании нет (строка таблицы не заполнялась).
            "spine": [int(v) if v is not None else None for v in (row[2], row[3], row[4])],
            "hip_right": [int(v) if v is not None else None for v in (row[5], row[6])],
            "hip_left": [int(v) if v is not None else None for v in (row[7], row[8])],
            # Итоговые классы качества («Итог» в таблице).
            "result": {"spine": row[9], "hip_right": row[10], "hip_left": row[11]},
            "comment": str(row[12]).strip() if row[12] else "",
        }
    return labels


def build_records(data_root: Path, labels: dict[int, dict]) -> list[dict]:
    """Соединяет DICOM-файлы с метками; возвращает список записей по изображениям.

    y_true берётся из «Итога» таблицы; строки, где единственной причиной
    итога=1 была клиника (все технические флаги 0 + клинический комментарий),
    корректируются до y=0 — см. докстринг скрипта.
    """
    records = []
    for num in sorted(labels):
        folder = data_root / str(num)
        dcm_files = sorted(folder.rglob("*.dcm")) if folder.exists() else []
        is_clinical, comment = classify_comment(labels[num]["comment"])
        # Части тела, у которых в исследовании ЕСТЬ и файл, и разметка: клинический
        # комментарий привязывается к ним (упомянутая часть может иметь файл, но
        # не иметь разметки — например, эндопротез правого бедра при №66: снимок
        # есть, но денситометрию бедру не выполняли, строки в таблице нет; тогда
        # клиника видна на снимке позвоночника и привязывается к нему).
        marked_parts = set()
        for path in dcm_files:
            part = PART_BY_FILENAME.get(path.name)
            if part is not None and not any(v is None for v in labels[num][part]):
                marked_parts.add(part)
        for path in dcm_files:
            part = PART_BY_FILENAME.get(path.name)
            if part is None:
                continue  # служебные файлы других серий не рассматриваем
            flags = labels[num][part]
            if any(v is None for v in flags):
                continue  # снимок без разметки данной части — в обучение не входит
            y = int(labels[num]["result"][part])  # эталонный класс качества
            # Коррекция «клинического итога»: итог=1 при всех флагах 0 и
            # клиническом комментарии -> нарушения качества нет.
            if y == 1 and max(flags) == 0 and is_clinical:
                y = 0
            clinical_flag, clinical_text = clinical_targets(part, is_clinical, comment, marked_parts)
            records.append({
                "study": num,
                "path": path,
                "part": part,
                "y": y,
                "flags": flags,
                "clinical": int(clinical_flag),
                "clinical_text": clinical_text,
                "comment": labels[num]["comment"],
            })
    return records


# ----------------------------------------------------------------------------
# 3. Случайное перемешивание и разбиение на обучение/валидацию
# ----------------------------------------------------------------------------

def shuffle_split(records: list[dict], seed: int, val_share: float = 0.2) -> list[dict]:
    """Перемешивает ИССЛЕДОВАНИЯ в случайном порядке и делит 80/20.

    Разбиение выполняется по исследованиям, а не по отдельным снимкам: снимки
    одного пациента коррелированы, и их попадание одновременно в обучение и
    валидацию завысило бы метрики (утечка данных).
    """
    studies = sorted({r["study"] for r in records})
    rng = random.Random(seed)
    for attempt in range(10):  # добиваемся, чтобы в валидации были оба класса
        shuffled = studies.copy()
        rng.shuffle(shuffled)  # случайный порядок исследований
        val_studies = set(shuffled[: round(len(shuffled) * val_share)])
        for r in records:
            r["split"] = "val" if r["study"] in val_studies else "train"
        val_ys = {r["y"] for r in records if r["split"] == "val"}
        train_ys = {r["y"] for r in records if r["split"] == "train"}
        if val_ys == {0, 1} and train_ys == {0, 1}:
            break
        seed += 1  # крайне маловероятный вырожденный сплит — перемешиваем заново
    return records


# ----------------------------------------------------------------------------
# 4. Датасет и аугментации
# ----------------------------------------------------------------------------

def image_transform(training: bool):
    """Пайплайн предобработки: [0..1] -> тензор 3x224x224 под ResNet18.

    При обучении добавляются малые поворот/сдвиг/масштаб — имитация вариабельности
    позиционирования (методичка п. 2.5-2.7: укладка варьирует от пациента к пациенту).
    """
    ops = [transforms.ToPILImage(), transforms.Resize((224, 224))]
    if training:
        ops += [transforms.RandomRotation(3),
                transforms.RandomAffine(0, translate=(0.02, 0.02), scale=(0.97, 1.03))]
    ops += [transforms.Grayscale(num_output_channels=3),  # ResNet ожидает 3 канала
            transforms.ToTensor(),
            # нормализация по статистике ImageNet — как у предобученной магистрали
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]
    return transforms.Compose(ops)


class CalibrationDataset(Dataset):
    """Читает DICOM при каждом обращении (файлы маленькие, кэш не нужен)."""

    def __init__(self, records: list[dict], training: bool):
        self.records = records
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        pixels = normalized_pixels(read_dataset(r["path"]))  # [0..1], float32
        violation = np.zeros(len(VIOLATIONS), dtype=np.float32)
        for local_i, global_i in enumerate(PART_VIOLATION_MASK[r["part"]]):
            violation[global_i] = r["flags"][local_i]  # метки применимой части
        return (
            self.transform(pixels),
            torch.tensor({"spine": 0, "hip_right": 1, "hip_left": 2}[r["part"]], dtype=torch.long),
            torch.tensor(r["y"], dtype=torch.float32),        # y_true качества
            torch.from_numpy(violation),                      # y_true нарушений
            torch.tensor(r["clinical"], dtype=torch.float32),  # y_true клиники
        )


# ----------------------------------------------------------------------------
# 5. Модель и угол оси
# ----------------------------------------------------------------------------
# CalibrationModel (ResNet18 + 4 головы) и spine_axis_angle (угол между
# перпендикуляром к оси сканирования, опущенным к остистому отростку нижнего
# поясничного позвонка, и прямой через остистые отростки первого и последнего
# позвонков) импортируются из dxa_qc.calibration — это тот же код, которым
# пользуется веб-приложение после обучения.


# ----------------------------------------------------------------------------
# 6. Обучение
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device):
    """Вероятности всех голов на выборке + метрики бинарного качества."""
    model.eval()
    part_pred, quality_p, viol_p, clin_p = [], [], [], []
    part_true, quality_true, viol_true, clin_true = [], [], [], []
    for images, part, y, violation, clinical in loader:
        out_part, out_q, out_v, out_c = model(images.to(device))
        part_pred.append(out_part.argmax(1).cpu())
        part_true.append(part)
        quality_p.append(torch.sigmoid(out_q).cpu())
        quality_true.append(y)
        viol_p.append(torch.sigmoid(out_v).cpu())
        viol_true.append(violation)
        clin_p.append(torch.sigmoid(out_c).cpu())
        clin_true.append(clinical)
    quality_p = torch.cat(quality_p).numpy()
    quality_true = torch.cat(quality_true).numpy()
    pred = (quality_p >= 0.5).astype(int)
    tp = int(((quality_true == 1) & (pred == 1)).sum())
    fn = int(((quality_true == 1) & (pred == 0)).sum())
    tn = int(((quality_true == 0) & (pred == 0)).sum())
    fp = int(((quality_true == 0) & (pred == 1)).sum())
    order = np.argsort(quality_p)  # ROC AUC по рангам (Манн-Уитни)
    ranks = np.empty(len(quality_p)); ranks[order] = np.arange(1, len(quality_p) + 1)
    pos, neg = int(quality_true.sum()), int((1 - quality_true).sum())
    auc = float((ranks[quality_true == 1].sum() - pos * (pos + 1) / 2) / max(pos * neg, 1))
    return {
        "prob": quality_p,
        "true": quality_true,
        "metrics": {
            "sensitivity": round(tp / max(tp + fn, 1), 4),
            "specificity": round(tn / max(tn + fp, 1), 4),
            "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
            "roc_auc": round(auc, 4),
            "part_accuracy": round(float((torch.cat(part_pred) == torch.cat(part_true)).float().mean()), 4),
            "support": len(quality_true),
        },
        "violation_prob": torch.cat(viol_p).numpy(),
        "violation_true": torch.cat(viol_true).numpy(),
        "clinical_prob": torch.cat(clin_p).numpy(),
        "clinical_true": torch.cat(clin_true).numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Калибровочное обучение DXA QC")
    # По умолчанию — папка ОБУЧАЮЩИХ исследований (80 шт., без тестовых):
    # тестовый набор «Исследования_тест» обслуживает сайт и при дообучении
    # должен оставаться невидимым для модели.
    parser.add_argument("--data-root", type=Path, default=Path("Исследования_обучение"))
    parser.add_argument("--labels", type=Path,
                        default=Path(r"C:\Users\smorz\Downloads\Telegram Desktop\разметка.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Воспроизводимость: фиксируем все источники случайности.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")  # GPU не требуется: данные небольшие

    # --- Данные ---
    labels = read_labels(args.labels)
    records = build_records(args.data_root, labels)
    records = shuffle_split(records, args.seed)
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    print(f"Снимков с разметкой: {len(records)} | train: {len(train)} | val: {len(val)}")
    print("train классы:", dict(Counter(r["y"] for r in train)),
          "| val классы:", dict(Counter(r["y"] for r in val)))
    print("train исследования:", len({r["study"] for r in train}),
          "| val исследования:", len({r["study"] for r in val}))

    # --- Загрузчики ---
    loaders = {
        "train": DataLoader(CalibrationDataset(train, True), batch_size=args.batch_size,
                            shuffle=True),
        "val": DataLoader(CalibrationDataset(val, False), batch_size=args.batch_size),
    }

    # --- Модель и лоссы ---
    # Инициализация магистрали доменными весами (см. CalibrationModel): модель
    # проекта уже обучена на DXA-снимках того же формата, что калибровочные.
    model = CalibrationModel(domain_checkpoint=args.model_dir / "dxa_qc_resnet18.pt").to(device)
    # Финальная конфигурация подобрана тремя прогонами:
    #   (1) ImageNet + полное файнтюнирование lr=3e-4  -> F1 0.51, AUC 0.58;
    #   (2) ImageNet + только layer4                  -> F1 0.52, AUC 0.60;
    #   (3) доменные веса + полное FT lr=1e-4 + cosine -> F1 0.51, AUC 0.58;
    #   (4) доменные веса + только layer4 (ниже)       -> F1 0.53, AUC 0.61 <- лучшая.
    # Признаки доменной магистрали уже релевантны DXA, поэтому обучаем только
    # последний блок и головы: полное файнтюнирование на ~200 снимках
    # переобучается и выигрыша не даёт.
    for name, parameter in model.backbone.named_parameters():
        if not name.startswith("layer4"):
            parameter.requires_grad = False
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    # Балансировка классов: «годных» больше, чем нарушений — pos_weight в BCE.
    positives = sum(r["y"] for r in train)
    quality_weight = torch.tensor((len(train) - positives) / max(positives, 1))
    clin_pos = sum(r["clinical"] for r in train)
    clinical_weight = torch.tensor((len(train) - clin_pos) / max(clin_pos, 1))
    ce_part = nn.CrossEntropyLoss()
    bce_quality = nn.BCEWithLogitsLoss(pos_weight=quality_weight)
    bce_clinical = nn.BCEWithLogitsLoss(pos_weight=clinical_weight)

    # Маски применимости нарушений для батча: лосс по «чужим» меткам не считается.
    def violation_loss(logits, targets, parts):
        loss, count = 0.0, 0
        probs = torch.sigmoid(logits)
        for i, part in enumerate(parts):
            idx = PART_VIOLATION_MASK[part]
            loss = loss + F.binary_cross_entropy(
                probs[i, idx], targets[i, idx], reduction="mean")
            count += 1
        return loss / max(count, 1)

    # --- Цикл обучения с ранней остановкой по F1 качества на валидации ---
    # После заморозки ранних слоёв обучается меньше параметров -> можно
    # учить дольше и терпеливее ждать улучшений (patience 6 вместо 3).
    STALE_LIMIT = 6
    best_f1, best_state, stale = -1.0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, part, y, violation, clinical in loaders["train"]:
            images, part, y, violation, clinical = (
                images.to(device), part.to(device), y.to(device),
                violation.to(device), clinical.to(device))
            optimizer.zero_grad(set_to_none=True)
            out_part, out_q, out_v, out_c = model(images)
            parts = ["spine" if p == 0 else "hip_right" if p == 1 else "hip_left"
                     for p in part.tolist()]
            # Суммарный лосс: часть + качество(y из «Итога») + нарушения + клиника.
            loss = (ce_part(out_part, part)
                    + bce_quality(out_q, y)
                    + violation_loss(out_v, violation, parts)
                    + bce_clinical(out_c, clinical))
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
        result = evaluate(model, loaders["val"], device)
        history.append({"epoch": epoch,
                        "train_loss": round(total / len(train), 5),
                        "val": result["metrics"]})
        print(history[-1])
        if result["metrics"]["f1"] > best_f1:
            best_f1, stale = result["metrics"]["f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= STALE_LIMIT:  # ранняя остановка: 6 эпох без улучшения
                break
    model.load_state_dict(best_state)  # откат к лучшей эпохе

    # --- Итоговая оценка и отчёт по каждому снимку ---
    result = evaluate(model, loaders["val"], device)
    print("\nИтоговые метрики на валидации:", json.dumps(result["metrics"], ensure_ascii=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    # Сохранившееся разбиение — для воспроизводимости сравнения моделей.
    with (args.output_dir / "calibration_split.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["study", "file", "part", "split"])
        for r in records:
            writer.writerow([r["study"], r["path"].name, r["part"], r["split"]])

    # Чекпойнт: веса + таксономия (инференс без обращения к таблице).
    torch.save({"model_state": best_state, "violations": VIOLATIONS,
                "part_titles": PART_TITLES},
               args.model_dir / "calibration_resnet18.pt")

    # Полный отчёт: по каждому снимку train+val, включая ВАЛИДАЦИОННЫЕ прогнозы.
    val_loader = DataLoader(CalibrationDataset(val, False), batch_size=args.batch_size)
    started = time.perf_counter()
    rows = []
    vi = 0  # индекс в валидационных массивах evaluate()
    for r in records:
        if r["split"] == "val":
            p_quality = float(result["prob"][vi])
            p_clinical = float(result["clinical_prob"][vi])
            vprob = result["violation_prob"][vi]
            vi += 1
        else:  # для train показываем эталон; прогноз модели не считается (in-sample)
            p_quality, p_clinical, vprob = None, None, None
        angle = deviation = None
        angle_ok = ""
        if r["part"] == "spine":  # угол считается только для позвоночника
            image = normalized_pixels(read_dataset(r["path"]))
            angle, deviation, _ = spine_axis_angle(image)
            if not np.isnan(angle):
                angle_ok = "да" if deviation <= AXIS_LIMIT_DEG else "нет"
        applicable = PART_VIOLATION_MASK[r["part"]]
        flags_true = ";".join(VIOLATIONS[g] for g, local in zip(applicable, r["flags"]) if local)
        flags_pred = ""
        if vprob is not None:
            flags_pred = ";".join(VIOLATIONS[g] for g in applicable if vprob[g] >= 0.5) or "none"
        rows.append({
            "исследование": r["study"],
            "файл": r["path"].name,
            "часть тела": PART_TITLES[r["part"]],
            "выборка": "train" if r["split"] == "train" else "validation",
            "y_эталон (из таблицы)": r["y"],
            "y_прогноз": int(p_quality >= 0.5) if p_quality is not None else "",
            "P(нарушение качества)": round(p_quality, 3) if p_quality is not None else "",
            "нарушения_эталон": flags_true or "none",
            "нарушения_прогноз": flags_pred,
            "угол_построения_град": round(angle, 2) if angle is not None and not np.isnan(angle) else "",
            "отклонение_оси_град": round(deviation, 2) if deviation is not None and not np.isnan(deviation) else "",
            f"ось_выровнена_(до_{AXIS_LIMIT_DEG:.0f}гр)": angle_ok,
            # ОТДЕЛЬНАЯ ячейка для клинических отклонений (не является нарушением снимка):
            "клиническое_отклонение": r["clinical_text"],
            "P(клиническое)": round(p_clinical, 3) if p_clinical is not None else "",
            "комментарий_таблицы": r["comment"],
        })
    columns = list(rows[0].keys())
    with (args.output_dir / "calibration_report.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {"model": "ResNet18 multitask (калибровка)", "seed": args.seed,
               "data": {"train": len(train), "validation": len(val),
                        "train_studies": len({r["study"] for r in train}),
                        "val_studies": len({r["study"] for r in val})},
               "validation": result["metrics"], "history": history,
               "axis_rule": f"отклонение оси до {AXIS_LIMIT_DEG:.0f}° по таблице разметки",
               "limitations": "Модель обучена на калибровочной разметке одного эксперта; "
                              "клинические отклонения (сколиоз, перелом, эндопротез, "
                              "люмбализация) не считаются нарушением качества снимка."}
    (args.model_dir / "calibration_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Отчёт: {args.output_dir / 'calibration_report.csv'}")
    print(f"Модель: {args.model_dir / 'calibration_resnet18.pt'}")
    print(f"Время инференса валидации: {time.perf_counter() - started:.1f} с")


if __name__ == "__main__":
    main()
