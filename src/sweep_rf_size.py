"""Size/accuracy sweep for the delivered Random Forest.

The forest is large because `min_samples_leaf` defaults to 1, so every tree
grows until its leaves are pure -- on 167K training rows that produces tens of
thousands of nodes per tree. This sweep asks whether a shallower forest gives
up any accuracy.

Selection is done on the **validation** split. The test split is not consulted
here; it is scored once, afterwards, for the configuration chosen on val.

Usage:
    python sweep_rf_size.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_loading import load_config
from evaluate import compute_classification_metrics
import model_ml as mlm
from preprocessing import patch_ndvi_series

LEAF_VALUES = [1, 2, 5, 10, 20]


def model_size_mb(model, extra: dict) -> float:
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        tmp = Path(f.name)
    joblib.dump({"model": model, "extra": extra}, tmp, compress=6)
    mb = tmp.stat().st_size / 1048576
    tmp.unlink()
    return mb


def main() -> None:
    cfg = load_config()
    train_ids = mlm.read_split(cfg, "train")
    val_ids = mlm.read_split(cfg, "val")
    ignore = cfg["evaluation"]["ignore_classes"]

    print(f"building NDVI feature table ({len(train_ids)} train patches)...")
    df = mlm.build_training_table(cfg, train_ids, feature_fn=patch_ndvi_series)
    cols = mlm.feature_columns(df)
    X, y = df[cols].values, df["label"].values

    base = dict(cfg["model_ndvi"]["random_forest"])
    rows = []
    for leaf in LEAF_VALUES:
        params = {**base, "min_samples_leaf": leaf}
        t0 = time.time()
        clf = RandomForestClassifier(**params)
        clf.fit(X, y)
        fit_s = time.time() - t0

        size = model_size_mb(clf, {"feature_names": cols})
        y_true, y_pred, _ = mlm.evaluate_full_patches(
            cfg, clf.predict, cols, val_ids, cfg["void_class"], feature_fn=patch_ndvi_series
        )
        m = compute_classification_metrics(y_true, y_pred, cfg["classes"], ignore_classes=ignore)
        n_nodes = int(np.sum([t.tree_.node_count for t in clf.estimators_]))
        rows.append({
            "min_samples_leaf": leaf, "size_mb": round(size, 1), "total_nodes": n_nodes,
            "val_accuracy": round(m["accuracy"], 4), "val_macro_f1": round(m["macro_f1"], 4),
            "fit_s": round(fit_s, 1),
        })
        print(f"  leaf={leaf:<3} {size:6.1f} MB  {n_nodes:>9,} nodes  "
              f"val_acc={m['accuracy']:.4f}  val_macroF1={m['macro_f1']:.4f}  ({fit_s:.0f}s)")

    out = pd.DataFrame(rows)
    path = Path(__file__).resolve().parent.parent / "outputs" / "metrics" / "rf_size_sweep_val.csv"
    out.to_csv(path, index=False)
    print(f"\n{out.to_string(index=False)}\nwrote {path}")

    best = out.loc[out["val_macro_f1"].idxmax()]
    print(f"\nbest val macro-F1: min_samples_leaf={int(best['min_samples_leaf'])} "
          f"({best['val_macro_f1']:.4f}, {best['size_mb']} MB)")


if __name__ == "__main__":
    main()
