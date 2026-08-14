"""Traditional-ML track: per-pixel Random Forest / XGBoost crop classifiers.

Trained on the stratified-sampled temporal-statistics feature table from
`preprocessing.build_pixel_feature_table` (per assignment task D). Validation
and test evaluation instead run on *every* labeled pixel of the held-out
patches (not just the sampled subset) so reported metrics reflect real
full-image performance.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from data_loading import PROJECT_ROOT, load_target
from preprocessing import build_pixel_feature_table, patch_ndvi_series, patch_pixel_features

NON_FEATURE_COLS = {"label", "patch_id", "row", "col"}


def read_split(config: dict, name: str) -> list[int]:
    path = PROJECT_ROOT / config["split"]["splits_dir"] / f"{name}.txt"
    return [int(line) for line in path.read_text().splitlines() if line.strip()]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def build_training_table(config: dict, patch_ids: list[int], feature_fn=patch_pixel_features) -> pd.DataFrame:
    return build_pixel_feature_table(
        config, patch_ids, seed=config["split"]["seed"], feature_fn=feature_fn
    )


def train_random_forest(X: np.ndarray, y: np.ndarray, config: dict) -> RandomForestClassifier:
    params = config["model_ml"]["random_forest"]
    clf = RandomForestClassifier(**params)
    clf.fit(X, y)
    return clf


def train_xgboost(X: np.ndarray, y: np.ndarray, config: dict) -> tuple[XGBClassifier, dict]:
    """XGBoost needs contiguous 0..K-1 labels; we remap and return the mapping."""
    classes = sorted(np.unique(y).tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    y_idx = np.array([class_to_idx[c] for c in y])

    params = config["model_ml"]["xgboost"]
    clf = XGBClassifier(
        **params,
        num_class=len(classes),
        objective="multi:softmax",
        eval_metric="mlogloss",
    )
    # XGBoost has no built-in `class_weight='balanced'`; approximate it with
    # per-sample weights inversely proportional to class frequency, matching
    # the Random Forest's class-imbalance handling for a fair comparison.
    counts = pd.Series(y_idx).value_counts()
    weights = pd.Series(y_idx).map(lambda c: len(y_idx) / (len(counts) * counts[c])).values
    clf.fit(X, y_idx, sample_weight=weights)
    return clf, idx_to_class


def predict_xgboost(clf: XGBClassifier, X: np.ndarray, idx_to_class: dict) -> np.ndarray:
    pred_idx = clf.predict(X)
    return np.array([idx_to_class[i] for i in pred_idx])


def evaluate_full_patches(
    config: dict,
    predict_fn,
    feature_names: list[str],
    patch_ids: list[int],
    void_class: int,
    feature_fn=patch_pixel_features,
) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Run predict_fn over every labeled pixel of each patch.

    predict_fn: callable(X: (N, F) array) -> (N,) predicted class IDs.
    feature_fn must match the one used to build the training table.
    Returns (y_true_flat, y_pred_flat, {patch_id: predicted_label_map (H,W)}).
    """
    y_true_all, y_pred_all = [], []
    pred_maps = {}
    for pid in patch_ids:
        features, names = feature_fn(config, pid)  # (F, H, W)
        assert names == feature_names, "Feature order mismatch between train and eval"
        target = load_target(config, pid)  # (H, W)

        F, H, W = features.shape
        X = features.reshape(F, -1).T  # (H*W, F)
        y_pred_flat = predict_fn(X)
        pred_map = y_pred_flat.reshape(H, W)
        pred_maps[pid] = pred_map

        mask = target.reshape(-1) != void_class
        y_true_all.append(target.reshape(-1)[mask])
        y_pred_all.append(y_pred_flat[mask])

    return np.concatenate(y_true_all), np.concatenate(y_pred_all), pred_maps


def save_model(model, path: str | Path, extra: dict | None = None, compress: int = 6) -> None:
    """Persist a fitted model plus the metadata needed to use it again.

    `extra` must carry `feature_names` (and, for XGBoost, `idx_to_class`):
    without them a loaded model cannot be applied safely, since feature order
    and label encoding are not recoverable from the estimator alone.

    Compressed by default -- an unpruned 300-tree forest serialises to ~775 MB
    raw but ~90 MB at compress=6, which is the difference between an artifact
    that can be attached to a release and one that cannot.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "extra": extra or {}}, path, compress=compress)


def load_model(path: str | Path) -> tuple[object, dict]:
    obj = joblib.load(path)
    return obj["model"], obj["extra"]
