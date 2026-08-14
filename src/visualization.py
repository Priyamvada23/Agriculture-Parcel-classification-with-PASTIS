"""Visualization utilities for model outputs (assignment task F).

Produces, for selected test patches: Sentinel-2 RGB | ground truth | model
prediction panels (one row per model, so ML and DL predictions can be
compared side by side on the same patches), plus confusion-matrix heatmaps
and the class-distribution chart used in the EDA.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loading import PROJECT_ROOT, load_config, load_s2, load_target
from preprocessing import to_reflectance

# Fixed colormap so the same class always gets the same color across every
# figure (GT vs. predictions, different models, different patches).
_CLASS_CMAP = plt.cm.tab20
_CLASS_NORM = mcolors.Normalize(vmin=0, vmax=19)


def _label_map(ax, arr, title):
    ax.imshow(arr, cmap=_CLASS_CMAP, norm=_CLASS_NORM, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _rgb(config, patch_id, timestep=None):
    s2 = load_s2(config, patch_id)
    refl = to_reflectance(s2)
    t = timestep if timestep is not None else s2.shape[0] // 2
    rgb = refl[t][config["rgb_bands"]].transpose(1, 2, 0)
    return np.clip(rgb / 0.3, 0, 1)


def plot_prediction_panels(
    config: dict,
    patch_id: int,
    predictions: dict[str, np.ndarray],
    out_path: str | Path,
) -> None:
    """One row: RGB | ground truth | prediction(s), for a single patch.

    predictions: {model_display_name: predicted_label_map (H, W)}
    """
    target = load_target(config, patch_id)
    n_cols = 2 + len(predictions)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.6))

    axes[0].imshow(_rgb(config, patch_id))
    axes[0].set_title(f"Patch {patch_id}\nSentinel-2 RGB", fontsize=10)
    axes[0].axis("off")

    _label_map(axes[1], target, "Ground truth")

    for i, (name, pred) in enumerate(predictions.items()):
        _label_map(axes[2 + i], pred, f"Prediction: {name}")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(
    cm_normalized: np.ndarray,
    label_names: list[str],
    title: str,
    out_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized fraction")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


def load_test_predictions(config: dict) -> dict[str, dict[int, np.ndarray]]:
    """Load every outputs/predictions/*_test_predictions.npz found, keyed by model tag."""
    pred_dir = PROJECT_ROOT / config["output"]["predictions_dir"]
    result = {}
    for path in pred_dir.glob("*_test_predictions.npz"):
        tag = path.stem.replace("_test_predictions", "")
        if tag in EXCLUDED_TAGS:
            continue
        with np.load(path) as npz:
            result[tag] = {int(k): npz[k] for k in npz.files}
    return result


MODEL_DISPLAY_NAMES = {
    "ml_rf": "RF (temporal stats)",
    "ml_xgb": "XGBoost (temporal stats)",
    "ndvi_rf": "RF (NDVI series)",
    "ndvi_dl": "TempCNN (NDVI series)",
    "dl": "Temporal-Attn U-Net",
}

# Superseded runs kept on disk for reference but excluded from figures, so
# panels show one column per *current* model. `dl10ep` is the earlier
# 10-epoch U-Net, retained only for the epochs-vs-performance comparison in
# the report; the 40-epoch run under `dl` replaces it.
EXCLUDED_TAGS = {"dl10ep"}


def build_all_prediction_panels(config: dict, n_patches: int = 4) -> list[Path]:
    all_preds = load_test_predictions(config)
    if not all_preds:
        print("No prediction files found under outputs/predictions/ — run train.py first.")
        return []

    common_patch_ids = set.intersection(*(set(d.keys()) for d in all_preds.values()))
    selected = sorted(common_patch_ids)[:n_patches]

    fig_dir = PROJECT_ROOT / config["output"]["figures_dir"]
    out_paths = []
    for pid in selected:
        preds = {MODEL_DISPLAY_NAMES.get(tag, tag): all_preds[tag][pid] for tag in sorted(all_preds)}
        out_path = fig_dir / f"prediction_panel_patch_{pid}.png"
        plot_prediction_panels(config, pid, preds, out_path)
        out_paths.append(out_path)
        print("saved", out_path)
    return out_paths


def build_all_confusion_matrices(config: dict) -> list[Path]:
    metrics_dir = PROJECT_ROOT / config["output"]["metrics_dir"]
    fig_dir = PROJECT_ROOT / config["output"]["figures_dir"]
    out_paths = []
    for path in metrics_dir.glob("*_test_confusion_matrix.csv"):
        tag = path.stem.replace("_test_confusion_matrix", "")
        if tag in EXCLUDED_TAGS:
            continue
        cm_df = pd.read_csv(path, index_col=0)
        cm = cm_df.values.astype(np.float64)
        cm_norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        out_path = fig_dir / f"confusion_matrix_{tag}_test.png"
        plot_confusion_matrix(
            cm_norm, list(cm_df.columns),
            f"{MODEL_DISPLAY_NAMES.get(tag, tag)} — test set confusion matrix (row-normalized)",
            out_path,
        )
        out_paths.append(out_path)
        print("saved", out_path)
    return out_paths


if __name__ == "__main__":
    cfg = load_config()
    build_all_confusion_matrices(cfg)
    build_all_prediction_panels(cfg, n_patches=4)
