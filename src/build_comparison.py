"""Build the cross-model comparison table from the saved metric files.

Reads `outputs/metrics/{tag}_{split}_summary.json` for every model and writes
a single tidy comparison CSV. Derived from the metric files rather than
recomputed, so it cannot drift from them.

Replaces the earlier `ml_model_comparison_test.csv`, which covered only the
two tree models and was written mid-run (so it went stale as soon as the
evaluation protocol changed).

Usage:
    python build_comparison.py                 # test split
    python build_comparison.py --split val
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from data_loading import PROJECT_ROOT, load_config

# tag -> (track, representation, model) for the report tables
MODELS = [
    ("ndvi_rf", "C (delivered)", "46 ordered NDVI values", "Random Forest"),
    ("ndvi_dl", "D",             "46 ordered NDVI values", "TempCNN (1D)"),
    ("ml_xgb",  "A",             "60 temporal statistics", "XGBoost"),
    ("ml_rf",   "A",             "60 temporal statistics", "Random Forest"),
    ("dl",      "B",             "raw (T,C,H,W), 10 bands", "Temporal-Attn U-Net"),
]


def build(config: dict, split: str) -> pd.DataFrame:
    mdir = PROJECT_ROOT / config["output"]["metrics_dir"]
    rows = []
    for tag, track, features, model in MODELS:
        primary = mdir / f"{tag}_{split}_summary.json"
        if not primary.exists():
            print(f"  (skipping {tag}: no {split} metrics)")
            continue
        s = json.loads(primary.read_text())
        per = pd.read_csv(mdir / f"{tag}_{split}_per_class.csv")
        row = {
            "track": track, "features": features, "model": model,
            "accuracy": round(s["accuracy"], 4),
            "macro_f1": round(s["macro_f1"], 4),
            "macro_f1_present_classes": round(per[per.support > 0]["f1"].mean(), 4),
            "weighted_f1": round(s["weighted_f1"], 4),
            "mean_iou": round(s["mean_iou"], 4),
        }
        ref = mdir / f"{tag}_{split}_withbg_summary.json"
        if ref.exists():
            r = json.loads(ref.read_text())
            row["accuracy_withbg"] = round(r["accuracy"], 4)
            row["macro_f1_withbg"] = round(r["macro_f1"], 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = build(cfg, args.split)
    out = PROJECT_ROOT / cfg["output"]["metrics_dir"] / f"model_comparison_{args.split}.csv"
    df.to_csv(out, index=False)
    pd.set_option("display.width", 190)
    print(f"\nModel comparison ({args.split} split, crop-only protocol):\n")
    print(df.to_string(index=False))
    print(f"\nwrote {out}")
