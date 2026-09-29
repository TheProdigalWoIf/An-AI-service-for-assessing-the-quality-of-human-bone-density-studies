#!/usr/bin/env python3
"""Обучение мультитаск-модели DXA QC на экспертной разметке.

Логика (схема экспертизы): снимок -> анатомическая область (lumbar_spine /
proximal_femur) -> сторона (none/right/left, голова обучения по экспертной
разметке) -> пять критериев (позвоночник: укладка, ось, артефакты; бедро:
ротация/позиционирование, ROI) -> ИТОГОВОЕ РЕШЕНИЕ.

Ключевые правила:
- итоговое качество НЕ обучается по колонке quality_class: плохой снимок =
  нарушен хотя бы один применимый критерий (derived_quality); колонка
  quality_class остаётся только маркером «строка размечена»;
- разделение train/validation = 85/15 ТОЛЬКО по study_uid: изображения одного
  исследования никогда не попадают в разные сплиты; тестовые снимки из Results
  НЕ выделяются — тестирование выполняется на внешней папке Исследования_тест;
- критерии чужой области исключаются из loss маской (masked violation loss);
- неразмеченные строки (quality_class пуст) и строки с exclude_from_training=True
  в обучение не попадают, но из данных не удаляются;
- пороги, ранняя остановка и метрики — по validation (оценка оптимистична,
  т.к. эта же выборка участвует в подборе порогов); финальная проверка —
  вручную на внешнем тесте Исследования_тест.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dxa_qc.dicom import normalized_pixels, read_dataset
from dxa_qc.metrics import binary_metrics, bootstrap_ci
from dxa_qc.ml import (
    DXAMultiTaskModel,
    REGIONS,
    VIOLATIONS,
    SIDES,
    allowed_violations,
    choose_device,
    image_transform,
    violation_vector,
)
from dxa_qc.repository import LabelStore, violation_codes


def resolve_image_path(root: Path, row) -> Path:
    """Путь к DICOM из строки разметки.

    path_to_study в CSV может быть записан упрощённо ('1\\1_1\\п.dcm'), а
    реальные файлы лежат глубже ('1\\series_002_2_CR\\A2504381177 DXA\\CR DXA\\п.dcm').
    Сначала пробуем прямое разрешение, затем ищем файл по имени внутри папки
    исследования (первый сегмент path_to_study либо study_uid).
    """
    relative = row.path_to_study.replace("\\", "/")
    candidates = []

    direct = root / relative
    if direct.is_file():
        return direct

    filename = relative.rsplit("/", 1)[-1]
    study_folder = relative.split("/", 1)[0]
    for folder in (root / study_folder, root / str(row.study_uid)):
        if folder.is_dir():
            candidates.extend(sorted(folder.rglob(filename)))

    unique = {path.resolve() for path in candidates}
    if len(unique) == 1:
        return unique.pop()
    if not unique:
        raise FileNotFoundError(f"Image not found for {row.path_to_study!r} under {root}")
    raise ValueError(f"Ambiguous path {row.path_to_study!r}: {sorted(map(str, unique))[:3]}")


def derived_quality(row) -> int:
    """Итоговое качество из критериев, а не из колонки quality_class.

    Плохой снимок = нарушен хотя бы один критерий, применимый к его области
    (схема экспертизы: ИТОГОВОЕ РЕШЕНИЕ формируется из ПРИЧИН). Колонка
    quality_class значениями в обучение не попадает.
    """
    allowed = allowed_violations(row.anatomical_region)
    return int(any(code in allowed for code in violation_codes(row)))


class DXADataset(Dataset):
    def __init__(self, rows, root: Path, training: bool):
        self.rows = rows
        self.root = root
        self.transform = image_transform(training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        image = normalized_pixels(read_dataset(resolve_image_path(self.root, row)))

        region = REGIONS.index(row.anatomical_region)
        side = SIDES.index(row.side)

        return (
            self.transform(image),
            torch.tensor(region, dtype=torch.long),
            torch.tensor(side, dtype=torch.long),
            torch.tensor(derived_quality(row), dtype=torch.float32),
            torch.from_numpy(violation_vector(row.violation_type)),
            row.image_uid,
            row.study_uid,
        )


def group_split(
    rows,
    seed: int,
    train_fraction: float = 0.85,
):
    """Разделение train/validation (85/15) по study_uid без утечки.

    Тестовые снимки из обучающей папки НЕ выделяются: роль теста выполняет
    внешняя неразмеченная папка Исследования_тест. Пропорции задаются по
    изображениям. Группы-исследования распределяются жадно (от больших к
    меньшим): каждую группу отдаём в тот сплит, который после этого меньше
    всего отклоняется и по размеру, и по доле плохих снимков (нарушен ≥1
    критерий, derived_quality) от целевых значений.
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")

    by_study = defaultdict(list)
    for row in rows:
        by_study[row.study_uid].append(row)

    if len(by_study) < 2:
        raise ValueError("At least two different study_uid groups are required")

    shares = {
        "train": train_fraction,
        "validation": 1.0 - train_fraction,
    }
    total_rows = len(rows)
    total_positive = sum(derived_quality(row) for row in rows)
    targets = {name: total_rows * share for name, share in shares.items()}
    positive_targets = {name: total_positive * share for name, share in shares.items()}

    groups = [
        {
            "rows": study_rows,
            "size": len(study_rows),
            "positive": sum(derived_quality(row) for row in study_rows),
        }
        for study_rows in by_study.values()
    ]
    random.Random(seed).shuffle(groups)
    groups.sort(key=lambda group: group["size"], reverse=True)

    result = {name: [] for name in shares}
    counts = Counter()
    positives = Counter()

    def total_error() -> float:
        return sum(
            abs(counts[name] - targets[name]) + abs(positives[name] - positive_targets[name])
            for name in shares
        )

    for group in groups:
        best_name, best_error = None, None
        for name in shares:
            result[name].extend(group["rows"])
            counts[name] += group["size"]
            positives[name] += group["positive"]
            error = total_error()
            counts[name] -= group["size"]
            positives[name] -= group["positive"]
            result[name] = result[name][: -group["size"]]
            if best_error is None or error < best_error:
                best_name, best_error = name, error

        result[best_name].extend(group["rows"])
        counts[best_name] += group["size"]
        positives[best_name] += group["positive"]

    if any(not result[name] for name in result):
        raise ValueError("Could not create non-empty train/validation splits")

    return result


def class_weights(rows, device):
    region_counts = Counter(row.anatomical_region for row in rows)
    region = torch.tensor(
        [len(rows) / max(2 * region_counts[name], 1) for name in REGIONS],
        dtype=torch.float32, device=device,
    )

    quality_positive = sum(derived_quality(row) for row in rows)
    quality = torch.tensor(
        (len(rows) - quality_positive) / max(quality_positive, 1),
        dtype=torch.float32, device=device,
    )

    matrix = np.stack([violation_vector(row.violation_type) for row in rows])
    positives = matrix.sum(axis=0)
    violation = torch.tensor(
        np.clip((len(rows) - positives) / np.maximum(positives, 1), 1, 12),
        dtype=torch.float32, device=device,
    )

    side_counts = Counter(row.side for row in rows)
    side = torch.tensor(
        [len(rows) / max(2 * side_counts[name], 1) for name in SIDES],
        dtype=torch.float32, device=device,
    )

    return region, side, quality, violation


def build_violation_mask(regions, device):
    """Маска [batch, критерии]: 1.0 = критерий применим к области строки."""
    masks = []
    for region_index in regions.detach().cpu().tolist():
        allowed = allowed_violations(REGIONS[int(region_index)])
        masks.append([1.0 if name in allowed else 0.0 for name in VIOLATIONS])
    return torch.tensor(masks, dtype=torch.float32, device=device)


def masked_violation_loss(violation_logits, violation_targets, regions, violation_weight, device):
    """BCE по критериям с учётом применимости к anatomical_region строки."""
    loss_per_violation = F.binary_cross_entropy_with_logits(
        violation_logits, violation_targets, pos_weight=violation_weight, reduction="none",
    )
    mask = build_violation_mask(regions, device)
    masked_loss = loss_per_violation * mask
    return masked_loss.sum() / mask.sum().clamp_min(1.0)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    region_logits, side_logits, quality_logits, violation_logits = [], [], [], []
    region_true, side_true, quality_true, violation_true = [], [], [], []
    image_uids, study_uids = [], []

    started = time.perf_counter()

    for images, region, side, quality, violation, image_uid, study_uid in loader:
        output = model(images.to(device))
        region_logits.append(output[0].cpu())
        side_logits.append(output[1].cpu())
        quality_logits.append(output[2].cpu())
        violation_logits.append(output[3].cpu())
        region_true.append(region)
        side_true.append(side)
        quality_true.append(quality)
        violation_true.append(violation)
        image_uids.extend(image_uid)
        study_uids.extend(study_uid)

    elapsed = time.perf_counter() - started

    return {
        "region_prob": torch.softmax(torch.cat(region_logits), dim=1).numpy(),
        "side_prob": torch.softmax(torch.cat(side_logits), dim=1).numpy(),
        "quality_prob": torch.sigmoid(torch.cat(quality_logits)).numpy(),
        "violation_prob": torch.sigmoid(torch.cat(violation_logits)).numpy(),
        "region_true": torch.cat(region_true).numpy(),
        "side_true": torch.cat(side_true).numpy(),
        "quality_true": torch.cat(quality_true).numpy(),
        "violation_true": torch.cat(violation_true).numpy(),
        "image_uids": image_uids,
        "study_uids": study_uids,
        "seconds_per_image": elapsed / max(len(image_uids), 1),
    }


def best_threshold(y_true, probabilities, min_positives=10, min_negatives=10):
    """Порог с лучшим F1 на validation; без достаточного support — 0.5.

    Тюнинг допускается только при поддержке >= 10 позитивов и негативов и
    только в диапазоне 0.35–0.65: на маленьком validation агрессивная
    настройка порогов сама становится источником ложных срабатываний
    (проверено: порог 0.28/0.31 с validation давал специфичность 0.11
    для ротации на test против 0.21 при 0.5 и худший F1 по качеству).
    """
    y_true = np.asarray(y_true, int)
    positives, negatives = int(y_true.sum()), int((1 - y_true).sum())
    if positives < min_positives or negatives < min_negatives:
        return 0.5
    candidates = np.arange(0.35, 0.66, 0.01)
    scores = [binary_metrics(y_true, probabilities, threshold)["f1"] for threshold in candidates]
    best = float(candidates[int(np.argmax(scores))])
    return best if binary_metrics(y_true, probabilities, best)["f1"] > binary_metrics(y_true, probabilities, 0.5)["f1"] + 0.05 else 0.5


def score_predictions(result) -> float:
    """Composite-скор для ранней остановки: среднее четырёх задач."""
    region_accuracy = float(np.mean(np.argmax(result["region_prob"], axis=1) == result["region_true"]))

    # Сторона значима только для бедра: у позвоночника она всегда none.
    femur = result["region_true"] == REGIONS.index("proximal_femur")
    side_accuracy = float(np.mean(
        np.argmax(result["side_prob"][femur], axis=1) == result["side_true"][femur]
    )) if femur.any() else 1.0

    quality = binary_metrics(result["quality_true"], result["quality_prob"], 0.5)["balanced_accuracy"]

    supported_f1 = []
    for index in range(len(VIOLATIONS)):
        support = int(result["violation_true"][:, index].sum())
        if support > 0:
            supported_f1.append(
                binary_metrics(result["violation_true"][:, index], result["violation_prob"][:, index], 0.5)["f1"]
            )

    violation_score = float(np.mean(supported_f1)) if supported_f1 else 0.0

    return (region_accuracy + side_accuracy + quality + violation_score) / 4.0


def build_metrics(result, quality_threshold, violation_thresholds):
    quality = binary_metrics(result["quality_true"], result["quality_prob"], quality_threshold)
    quality["f1_95_ci"] = bootstrap_ci(
        result["quality_true"], result["quality_prob"], result["study_uids"],
        lambda y, p: binary_metrics(y, p, quality_threshold)["f1"],
    )
    quality["roc_auc_95_ci"] = bootstrap_ci(
        result["quality_true"], result["quality_prob"], result["study_uids"],
        lambda y, p: binary_metrics(y, p, quality_threshold)["roc_auc"],
    )

    region_pred = np.argmax(result["region_prob"], axis=1)
    region_f1 = [
        binary_metrics(result["region_true"] == index, result["region_prob"][:, index])["f1"]
        for index in range(len(REGIONS))
    ]

    side_pred = np.argmax(result["side_prob"], axis=1)

    violation_metrics = {}
    for index, name in enumerate(VIOLATIONS):
        violation_metrics[name] = binary_metrics(
            result["violation_true"][:, index],
            result["violation_prob"][:, index],
            violation_thresholds[index],
        )
        violation_metrics[name]["support"] = int(result["violation_true"][:, index].sum())

    predicted = (result["violation_prob"] >= np.asarray(violation_thresholds)).astype(int)
    supported = [
        violation_metrics[name]["f1"]
        for name in VIOLATIONS
        if violation_metrics[name]["support"] > 0
    ]

    # Сторона: отдельно правое и левое бедро.
    side_metrics = {}
    for side_name, side_index in (("right", SIDES.index("right")), ("left", SIDES.index("left"))):
        mask = result["side_true"] == side_index
        side_metrics[side_name] = {
            "accuracy": round(float(np.mean(side_pred[mask] == result["side_true"][mask])), 4) if mask.any() else 0.0,
            "support": int(np.sum(mask)),
        }

    per_region_quality = {}
    for index, name in enumerate(REGIONS):
        mask = result["region_true"] == index
        per_region_quality[name] = binary_metrics(
            result["quality_true"][mask], result["quality_prob"][mask], quality_threshold,
        )
        per_region_quality[name]["images"] = int(mask.sum())

    return {
        "quality": quality,
        "quality_by_anatomical_region": per_region_quality,
        "region": {
            "accuracy": round(float(np.mean(region_pred == result["region_true"])), 4),
            "macro_f1": round(float(np.mean(region_f1)), 4),
        },
        "side": {
            "accuracy": round(float(np.mean(side_pred == result["side_true"])), 4),
            "right": side_metrics["right"],
            "left": side_metrics["left"],
        },
        "violations": {
            "macro_f1_supported_classes": round(float(np.mean(supported)), 4) if supported else 0.0,
            "unsupported_in_split": [
                name for name in VIOLATIONS if violation_metrics[name]["support"] == 0
            ],
            "exact_match": round(float(np.mean(np.all(predicted == result["violation_true"], axis=1))), 4),
            "per_class": violation_metrics,
        },
        "seconds_per_image": round(float(result["seconds_per_image"]), 4),
        "images": len(result["image_uids"]),
        "studies": len(set(result["study_uids"])),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a multitask ResNet18 for DXA QC")
    parser.add_argument("--data-root", type=Path, default=Path("Results"))
    parser.add_argument("--labels", type=Path, default=Path("annotations/labels.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    labels = LabelStore(args.labels).read()

    # В обучение идут только размеченные строки своей области без exclude-флага.
    # quality_class используется исключительно как маркер «строка размечена»:
    # целевые значения модель получает из критериев (derived_quality).
    skipped_unlabeled = sum(1 for row in labels if row.quality_class is None and not row.exclude_from_training)
    excluded = sum(1 for row in labels if row.exclude_from_training)
    rows = [
        row for row in labels
        if row.quality_class is not None
        and row.anatomical_region in REGIONS
        and not row.exclude_from_training
    ]

    print(f"Labels: {len(labels)} total, {excluded} excluded, "
          f"{skipped_unlabeled} unlabeled, {len(rows)} used for training")

    if len(rows) < 50:
        raise SystemExit("At least 50 labeled rows are required")

    # Пути проверяем до обучения: одна битая строка не должна ронять эпоху.
    root = args.data_root.resolve()
    unresolved = []
    for row in rows:
        try:
            resolve_image_path(root, row)
        except (FileNotFoundError, ValueError) as exc:
            unresolved.append(str(exc))
    if unresolved:
        for message in unresolved[:10]:
            print("UNRESOLVED:", message)
        raise SystemExit(f"{len(unresolved)} rows have unresolvable paths")

    splits = group_split(rows, seed=args.seed, train_fraction=0.85)

    for name, values in splits.items():
        classes = {derived_quality(row) for row in values}
        if classes != {0, 1}:
            raise SystemExit(f"Split {name} does not contain both quality classes: {classes}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "splits.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_uid", "study_uid", "split"])
        for name, values in splits.items():
            writer.writerows((row.image_uid, row.study_uid, name) for row in values)

    device = choose_device()

    loaders = {
        name: DataLoader(
            DXADataset(values, root, name == "train"),
            batch_size=args.batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, values in splits.items()
    }

    model = DXAMultiTaskModel(pretrained=True).to(device)

    region_weight, side_weight, quality_weight, violation_weight = class_weights(splits["train"], device)
    region_loss = nn.CrossEntropyLoss(weight=region_weight)
    side_loss = nn.CrossEntropyLoss(weight=side_weight)
    quality_loss = nn.BCEWithLogitsLoss(pos_weight=quality_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_score = -1.0
    best_state = None
    stale = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0

        for images, region, side, quality, violation, _, _ in loaders["train"]:
            images, region, side = images.to(device), region.to(device), side.to(device)
            quality, violation = quality.to(device), violation.to(device)

            optimizer.zero_grad(set_to_none=True)
            region_logits, side_logits, quality_logits, violation_logits = model(images)

            loss = (
                region_loss(region_logits, region)
                + side_loss(side_logits, side)
                + quality_loss(quality_logits, quality)
                + masked_violation_loss(violation_logits, violation, region, violation_weight, device)
            )
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)

        validation = predict(model, loaders["validation"], device)
        score = score_predictions(validation)
        history.append({
            "epoch": epoch,
            "train_loss": round(total / len(splits["train"]), 5),
            "validation_score": round(score, 5),
        })
        print(history[-1], flush=True)

        if score > best_score:
            best_score, stale = score, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}: no improvement for {args.patience} epochs.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")

    model.load_state_dict(best_state)

    # Пороги, ранняя остановка и метрики — по validation. Внутреннего test из
    # Results больше нет: финальная проверка модели — на внешней папке
    # Исследования_тест (вручную или scripts/batch_inference.py).
    validation = predict(model, loaders["validation"], device)
    quality_threshold = best_threshold(validation["quality_true"], validation["quality_prob"])
    violation_thresholds = [
        best_threshold(validation["violation_true"][:, i], validation["violation_prob"][:, i])
        if validation["violation_true"][:, i].sum() else 0.5
        for i in range(len(VIOLATIONS))
    ]

    metrics = {
        "model": "ResNet18 multitask (region/side/quality/5 criteria), ImageNet initialization",
        "device": str(device),
        "seed": args.seed,
        "split": {
            name: {
                "images": len(values),
                "studies": len({row.study_uid for row in values}),
            }
            for name, values in splits.items()
        },
        "labels": {
            "total": len(labels),
            "excluded": excluded,
            "unlabeled": skipped_unlabeled,
            "used": len(rows),
        },
        "thresholds": {
            "quality": quality_threshold,
            "violations": dict(zip(VIOLATIONS, violation_thresholds)),
        },
        "validation": build_metrics(validation, quality_threshold, violation_thresholds),
        "history": history,
        "limitations": (
            "Metrics are computed on the validation split, which is also used "
            "for early stopping and threshold tuning, so the estimate is "
            "optimistic; the real holdout is the external unlabeled folder "
            "Исследования_тест. Labels are non-clinical visual annotations "
            "and require physician verification. Support for several criteria "
            "is very small; per-class F1 for those criteria is not reliable."
        ),
    }

    checkpoint = {
        "model_state": best_state,
        "regions": REGIONS,
        "sides": SIDES,
        "violations": VIOLATIONS,
        "quality_threshold": quality_threshold,
        "violation_thresholds": violation_thresholds,
        "input_size": 224,
        "metrics": metrics["validation"],
    }
    torch.save(checkpoint, args.output_dir / "dxa_qc_resnet18.pt")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(json.dumps(metrics["validation"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
