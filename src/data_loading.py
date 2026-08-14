"""Data loading utilities for the PASTIS crop-type classification assignment.

Reads Sentinel-2 time-series patches (`DATA_S2/S2_<id>.npy`), their semantic
annotations (`ANNOTATIONS/TARGET_<id>.npy`), and patch metadata
(`metadata.geojson`) from the subset described in `configs/config.yaml`.
"""
from __future__ import annotations

import datetime as dt
from collections import Counter
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path = "configs/config.yaml") -> dict:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_metadata(config: dict) -> gpd.GeoDataFrame:
    """Load metadata.geojson as a GeoDataFrame, one row per patch."""
    path = PROJECT_ROOT / config["data"]["metadata_geojson"]
    gdf = gpd.read_file(path)
    gdf = gdf.sort_values("ID_PATCH").reset_index(drop=True)
    return gdf


def list_patch_ids(config: dict) -> list[int]:
    """Patch IDs available on disk (intersection of DATA_S2 and ANNOTATIONS)."""
    s2_dir = PROJECT_ROOT / config["data"]["s2_dir"]
    ann_dir = PROJECT_ROOT / config["data"]["annotations_dir"]
    s2_ids = {int(p.stem.split("_")[1]) for p in s2_dir.glob("S2_*.npy")}
    ann_ids = {int(p.stem.split("_")[1]) for p in ann_dir.glob("TARGET_*.npy")}
    common = sorted(s2_ids & ann_ids)
    missing_ann = s2_ids - ann_ids
    missing_s2 = ann_ids - s2_ids
    if missing_ann or missing_s2:
        raise ValueError(
            f"Patch ID mismatch between DATA_S2 and ANNOTATIONS: "
            f"{len(missing_ann)} S2 patches missing targets, "
            f"{len(missing_s2)} targets missing S2 files."
        )
    return common


def load_s2(config: dict, patch_id: int) -> np.ndarray:
    """Return the Sentinel-2 time series for a patch, shape (T, C, H, W)."""
    path = PROJECT_ROOT / config["data"]["s2_dir"] / f"S2_{patch_id}.npy"
    return np.load(path)


def load_target(config: dict, patch_id: int) -> np.ndarray:
    """Return the semantic label map for a patch, shape (H, W).

    The raw file has shape (1, H, W) — only the 0th (semantic) annotation
    layer is provided for this assignment — so we squeeze it here.
    """
    path = PROJECT_ROOT / config["data"]["annotations_dir"] / f"TARGET_{patch_id}.npy"
    arr = np.load(path)
    return arr[0]


def get_dates(gdf: gpd.GeoDataFrame, patch_id: int) -> list[dt.date]:
    """Acquisition dates for a patch's Sentinel-2 observations, in array order."""
    row = gdf.loc[gdf["ID_PATCH"] == patch_id].iloc[0]
    dates_map: dict = row["dates-S2"]
    ordered = sorted(dates_map.items(), key=lambda kv: int(kv[0]))
    return [dt.datetime.strptime(str(v), "%Y%m%d").date() for _, v in ordered]


def patch_ids_for_folds(gdf: gpd.GeoDataFrame, folds: Iterable[int]) -> list[int]:
    folds = set(folds)
    return sorted(gdf.loc[gdf["Fold"].isin(folds), "ID_PATCH"].astype(int).tolist())


def inspect_dataset(config: dict, gdf: gpd.GeoDataFrame | None = None) -> dict:
    """Summarize array shapes/dtypes and per-patch temporal length.

    Used by the EDA notebook (assignment task B: "Inspecting array dimensions
    and data types", "Number of temporal observations per patch").
    """
    if gdf is None:
        gdf = load_metadata(config)
    patch_ids = list_patch_ids(config)

    s2_shapes, s2_dtypes, target_shapes, target_dtypes, n_obs = set(), set(), set(), set(), {}
    for pid in patch_ids:
        s2 = load_s2(config, pid)
        tgt = load_target(config, pid)
        s2_shapes.add(s2.shape[1:])   # (C, H, W) — T varies, checked separately
        s2_dtypes.add(str(s2.dtype))
        target_shapes.add(tgt.shape)
        target_dtypes.add(str(tgt.dtype))
        n_obs[pid] = s2.shape[0]

    return {
        "n_patches": len(patch_ids),
        "patch_ids": patch_ids,
        "s2_spatial_band_shapes": s2_shapes,
        "s2_dtypes": s2_dtypes,
        "target_shapes": target_shapes,
        "target_dtypes": target_dtypes,
        "n_observations_per_patch": n_obs,
        "n_observations_min": min(n_obs.values()),
        "n_observations_max": max(n_obs.values()),
        "tiles": sorted(gdf["TILE"].unique().tolist()),
        "folds": dict(sorted(Counter(gdf["Fold"].tolist()).items())),
    }


def class_pixel_counts(config: dict, patch_ids: Iterable[int] | None = None) -> Counter:
    """Per-pixel class distribution across the given patches (default: all)."""
    if patch_ids is None:
        patch_ids = list_patch_ids(config)
    counts: Counter = Counter()
    for pid in patch_ids:
        tgt = load_target(config, pid)
        values, freqs = np.unique(tgt, return_counts=True)
        counts.update(dict(zip(values.tolist(), freqs.tolist())))
    return counts


if __name__ == "__main__":
    cfg = load_config()
    meta = load_metadata(cfg)
    summary = inspect_dataset(cfg, meta)
    for k, v in summary.items():
        if k == "n_observations_per_patch":
            continue
        print(f"{k}: {v}")
