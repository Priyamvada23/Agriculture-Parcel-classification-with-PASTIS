"""Recompute metrics for every saved prediction set, without retraining.

Model predictions are saved per patch in `outputs/predictions/*.npz`, so any
change to the *scoring* protocol -- which classes count, which are ignored --
can be applied to every model at once by re-scoring those stored predictions.
Retraining is unnecessary because the models themselves are unchanged.

Writes, for each model:
  {tag}_{split}_*            primary protocol   (evaluation.ignore_classes)
  {tag}_{split}_withbg_*     reference protocol (evaluation.reference_ignore_classes)

Usage:
    python recompute_metrics.py
"""
from __future__ import annotations

import numpy as np

from data_loading import PROJECT_ROOT, load_config, load_target
from evaluate import compute_classification_metrics, save_metrics
from model_ml import read_split

# Superseded runs kept on disk but not re-scored (see visualization.EXCLUDED_TAGS).
EXCLUDED_TAGS = {"dl10ep"}


def ground_truth(config: dict, patch_ids: list[int]) -> np.ndarray:
    return np.concatenate([load_target(config, p).reshape(-1) for p in patch_ids])


def recompute(config: dict) -> None:
    pred_dir = PROJECT_ROOT / config["output"]["predictions_dir"]
    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    class_names = config["classes"]
    primary = config["evaluation"]["ignore_classes"]
    reference = config["evaluation"]["reference_ignore_classes"]

    print(f"primary protocol   : ignore {primary} (crop pixels only)")
    print(f"reference protocol : ignore {reference} (background retained)\n")

    for path in sorted(pred_dir.glob("*_predictions.npz")):
        stem = path.stem.replace("_predictions", "")      # e.g. ndvi_rf_test
        tag, _, split = stem.rpartition("_")               # -> ("ndvi_rf", "_", "test")
        if tag in EXCLUDED_TAGS:
            continue

        patch_ids = read_split(config, split)
        y_true = ground_truth(config, patch_ids)
        with np.load(path) as npz:
            y_pred = np.concatenate([npz[str(p)].reshape(-1) for p in patch_ids])

        m = compute_classification_metrics(y_true, y_pred, class_names, ignore_classes=primary)
        save_metrics(m, metrics_dir, prefix=f"{tag}_{split}")

        m_ref = compute_classification_metrics(y_true, y_pred, class_names, ignore_classes=reference)
        save_metrics(m_ref, metrics_dir, prefix=f"{tag}_{split}_withbg")
        print()


def background_report(config: dict, tag: str = "ndvi_rf", split: str = "test") -> None:
    """Quantify the blind spot that crop-only scoring introduces."""
    patch_ids = read_split(config, split)
    y_true = ground_truth(config, patch_ids)
    path = PROJECT_ROOT / config["output"]["predictions_dir"] / f"{tag}_{split}_predictions.npz"
    with np.load(path) as npz:
        y_pred = np.concatenate([npz[str(p)].reshape(-1) for p in patch_ids])

    bg_class = config["background_class"]
    crop = ~np.isin(y_true, config["evaluation"]["ignore_classes"])
    bg = y_true == bg_class

    print(f"=== background interactions, {tag} on {split} ===")
    print(f"  crop pixels predicted as background : {(y_pred[crop] == bg_class).sum():,} "
          f"/ {crop.sum():,} ({(y_pred[crop] == bg_class).mean() * 100:.1f}%) -- still penalised")
    print(f"  background pixels predicted as crop : {(y_pred[bg] != bg_class).sum():,} "
          f"/ {bg.sum():,} ({(y_pred[bg] != bg_class).mean() * 100:.1f}%) -- NOT penalised")

    fp = y_pred[bg]
    fp = fp[fp != bg_class]
    vals, counts = np.unique(fp, return_counts=True)
    print("  background->crop errors land mostly on:")
    for i in np.argsort(-counts)[:5]:
        print(f"      {config['classes'][int(vals[i])]:28s} {counts[i]:>7,} px")


if __name__ == "__main__":
    cfg = load_config()
    recompute(cfg)
    background_report(cfg)
