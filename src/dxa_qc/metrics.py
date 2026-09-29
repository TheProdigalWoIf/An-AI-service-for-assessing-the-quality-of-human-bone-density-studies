from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred) -> tuple[int, int, int, int]:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def binary_metrics(y_true, probabilities, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, int)
    probabilities = np.asarray(probabilities, float)
    y_pred = (probabilities >= threshold).astype(int)
    tp, tn, fp, fn = confusion(y_true, y_pred)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round((sensitivity + specificity) / 2, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc(y_true, probabilities), 4),
    }


def roc_auc(y_true, scores) -> float:
    y_true, scores = np.asarray(y_true, int), np.asarray(scores, float)
    positives, negatives = int(y_true.sum()), int((1 - y_true).sum())
    if not positives or not negatives:
        return 0.5
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        indices = np.flatnonzero(scores == value)
        ranks[indices] = ranks[indices].mean()
    return float((ranks[y_true == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def macro_f1(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> float:
    values = [binary_metrics(y_true[:, index], probabilities[:, index], threshold)["f1"] for index in range(y_true.shape[1])]
    return round(float(np.mean(values)), 4)


def bootstrap_ci(y_true, probabilities, groups, metric, repetitions: int = 500, seed: int = 42) -> list[float]:
    y_true, probabilities, groups = np.asarray(y_true), np.asarray(probabilities), np.asarray(groups)
    unique = np.unique(groups)
    rng, values = np.random.default_rng(seed), []
    for _ in range(repetitions):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        values.append(float(metric(y_true[indices], probabilities[indices])))
    low, high = np.percentile(values, (2.5, 97.5))
    return [round(float(low), 4), round(float(high), 4)]
