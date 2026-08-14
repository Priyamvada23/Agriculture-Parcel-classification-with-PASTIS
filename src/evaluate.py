"""Shared evaluation metrics for both the traditional-ML and DL tracks.

Both tracks ultimately produce per-pixel (predicted_class, true_class) pairs,
so a single set of metric functions serves both — this keeps the ML vs. DL
comparison apples-to-apples (assignment task E / Track C of the spec).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: dict[int, str],
    ignore_classes: list[int] | None = None,
) -> dict:
    """Compute the full metric suite for one evaluation split.

    y_true/y_pred: 1D arrays of integer class IDs (flattened pixels).
    class_names: {class_id: name} for every class in the taxonomy.
    ignore_classes: class IDs excluded from all metrics (e.g. void=19).
    """
    ignore_classes = set(ignore_classes or [])
    keep = ~np.isin(y_true, list(ignore_classes))
    y_true, y_pred = y_true[keep], y_pred[keep]

    labels = sorted(c for c in class_names if c not in ignore_classes)
    label_names = [class_names[c] for c in labels]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class_id": labels,
            "class_name": label_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(np.float64) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)

    # Mean IoU over classes actually present in y_true or y_pred (avoids
    # diluting the mean with classes absent from this split entirely).
    ious = []
    for i, c in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        denom = tp + fp + fn
        if denom == 0:
            continue
        ious.append(tp / denom)
    mean_iou = float(np.mean(ious)) if ious else float("nan")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "mean_iou": mean_iou,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_normalized": cm_norm,
        "labels": labels,
        "label_names": label_names,
    }


def evaluate_and_save(
    config: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_maps: dict[int, np.ndarray],
    tag: str,
    split: str,
) -> dict:
    """Score one split under both protocols and persist metrics + predictions.

    Every track routes through this, so the evaluation protocol is defined in
    exactly one place and cannot drift between models. Returns the primary
    (crop-only) metrics dict.

    Writes:
      {tag}_{split}_*         primary   -- evaluation.ignore_classes  (crops only)
      {tag}_{split}_withbg_*  reference -- evaluation.reference_ignore_classes
      {tag}_{split}_predictions.npz     -- per-patch predicted label maps

    Predictions are saved for every split (not just test) so
    `recompute_metrics.py` can re-score any split if the protocol changes,
    without retraining.
    """
    from data_loading import PROJECT_ROOT

    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    predictions_dir = PROJECT_ROOT / config["output"]["predictions_dir"]
    predictions_dir.mkdir(parents=True, exist_ok=True)
    class_names = config["classes"]

    primary = compute_classification_metrics(
        y_true, y_pred, class_names, ignore_classes=config["evaluation"]["ignore_classes"]
    )
    save_metrics(primary, metrics_dir, prefix=f"{tag}_{split}")

    reference = compute_classification_metrics(
        y_true, y_pred, class_names,
        ignore_classes=config["evaluation"]["reference_ignore_classes"],
    )
    save_metrics(reference, metrics_dir, prefix=f"{tag}_{split}_withbg")

    if pred_maps:
        # Class IDs are 0-19, so uint8 is sufficient. Torch's argmax returns
        # int64, which would make these files 8x larger for no benefit.
        np.savez_compressed(
            predictions_dir / f"{tag}_{split}_predictions.npz",
            **{str(pid): np.asarray(arr, dtype=np.uint8) for pid, arr in pred_maps.items()},
        )
    return primary


def save_metrics(metrics: dict, out_dir: str | Path, prefix: str) -> None:
    """Write metrics to outputs/metrics/{prefix}_summary.json, _per_class.csv, _confusion_matrix.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "mean_iou": metrics["mean_iou"],
    }
    with open(out_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    metrics["per_class"].to_csv(out_dir / f"{prefix}_per_class.csv", index=False)

    cm_df = pd.DataFrame(
        metrics["confusion_matrix"], index=metrics["label_names"], columns=metrics["label_names"]
    )
    cm_df.to_csv(out_dir / f"{prefix}_confusion_matrix.csv")

    print(f"[{prefix}] accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f} "
          f"weighted_f1={summary['weighted_f1']:.4f} mean_iou={summary['mean_iou']:.4f}")
    print(f"  saved metrics to {out_dir}/{prefix}_*")
