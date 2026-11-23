"""Chapter 4 CN/MCI/AD metrics, including zero-denominator rules."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def classify_metrics(labels, probabilities):
    truth = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probabilities, dtype=np.float64)
    if probability.ndim != 2 or probability.shape != (len(truth), 3):
        raise ValueError("Expected one three-class probability vector per sample")
    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(truth, prediction, labels=[0, 1, 2])
    precision, recall, f1 = [], [], []
    for index in range(3):
        tp = float(matrix[index, index])
        predicted = float(matrix[:, index].sum())
        actual = float(matrix[index].sum())
        p = tp / predicted if predicted else 0.0
        r = tp / actual if actual else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    try:
        auc = float(roc_auc_score(truth, probability, labels=[0, 1, 2],
                                  multi_class="ovr", average="macro"))
    except ValueError:
        auc = None
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "macro_auc": auc,
        "confusion_matrix": matrix.tolist(),
    }


def five_model_summary(results):
    if len(results) != 5:
        raise ValueError("The thesis reports five independently trained models")
    summary = {}
    for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_auc"):
        values = [result[name] for result in results]
        if any(value is None for value in values):
            summary[name] = {"mean": None, "std": None}
        else:
            summary[name] = {"mean": float(np.mean(values)),
                             "std": float(np.std(values, ddof=1))}
    return summary
