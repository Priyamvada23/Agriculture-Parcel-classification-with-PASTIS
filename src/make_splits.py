"""Generate reproducible train/val/test patch-ID splits.

Uses PASTIS's official 5-fold spatial cross-validation assignment (the
`Fold` field in metadata.geojson) rather than a random shuffle, so that
patches from the same spatial neighborhood never straddle the train/test
boundary (avoids spatial autocorrelation leakage). Fold-to-split mapping is
set in configs/config.yaml under `split`.

Run: python src/make_splits.py
Writes: configs/splits/{train,val,test}.txt (one patch ID per line)
"""
from __future__ import annotations

from pathlib import Path

from data_loading import PROJECT_ROOT, load_config, load_metadata, patch_ids_for_folds


def make_splits(config: dict) -> dict[str, list[int]]:
    gdf = load_metadata(config)
    split_cfg = config["split"]

    splits = {
        "train": patch_ids_for_folds(gdf, split_cfg["train_folds"]),
        "val": patch_ids_for_folds(gdf, split_cfg["val_folds"]),
        "test": patch_ids_for_folds(gdf, split_cfg["test_folds"]),
    }

    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    overlaps = (
        (set(splits["train"]) & set(splits["val"]))
        | (set(splits["train"]) & set(splits["test"]))
        | (set(splits["val"]) & set(splits["test"]))
    )
    assert not overlaps, f"Split overlap detected: {overlaps}"
    assert len(all_ids) == len(gdf), (
        f"Splits cover {len(all_ids)} patches but metadata has {len(gdf)}; "
        "check that train/val/test folds partition all 5 folds."
    )
    return splits


def write_splits(config: dict, splits: dict[str, list[int]]) -> None:
    out_dir = PROJECT_ROOT / config["split"]["splits_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        path = out_dir / f"{name}.txt"
        path.write_text("\n".join(str(i) for i in ids) + "\n")
        print(f"{name}: {len(ids)} patches -> {path}")


if __name__ == "__main__":
    cfg = load_config()
    splits = make_splits(cfg)
    write_splits(cfg, splits)
