"""Feature engineering shared by the EDA notebook and the traditional-ML track.

Sentinel-2 L2A reflectance in the provided .npy files is scaled by 10000
(confirmed by inspecting DATA_S2/S2_30003.npy: dtype int16, range
[-105, 14924] — small negative values are normal atmospheric-correction
artifacts near zero reflectance). All functions here work in reflectance
units (raw / 10000.0) unless noted otherwise.

Band index reference (see configs/config.yaml `bands`):
  0 Blue(B2) 1 Green(B3) 2 Red(B4) 3 RE1(B5) 4 RE2(B6) 5 RE3(B7)
  6 NIR(B8)  7 NarrowNIR(B8A)  8 SWIR1(B11)  9 SWIR2(B12)
"""
from __future__ import annotations

from typing import Iterable

from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import PROJECT_ROOT, load_metadata, load_s2, load_target

BAND_RED, BAND_NIR, BAND_GREEN, BAND_SWIR1 = 2, 6, 1, 8
EPS = 1e-6


def to_reflectance(s2_raw: np.ndarray) -> np.ndarray:
    """Convert raw int16 DN array (T,C,H,W) to float reflectance."""
    return s2_raw.astype(np.float32) / 10000.0


def compute_ndvi(s2_refl: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Input/output shape (T, H, W)."""
    nir, red = s2_refl[:, BAND_NIR], s2_refl[:, BAND_RED]
    return (nir - red) / (nir + red + EPS)


def compute_ndwi(s2_refl: np.ndarray) -> np.ndarray:
    """Gao's NDWI (vegetation water content) = (NIR - SWIR1) / (NIR + SWIR1).

    Chosen over McFeeters' water-body NDWI (Green/NIR) because this is a
    crop-focused analysis: the NIR/SWIR1 formulation tracks canopy/soil
    moisture rather than open water, which is more informative per-crop here.
    Shape (T, H, W).
    """
    nir, swir1 = s2_refl[:, BAND_NIR], s2_refl[:, BAND_SWIR1]
    return (nir - swir1) / (nir + swir1 + EPS)


def _temporal_stats(arr_t_first: np.ndarray) -> dict[str, np.ndarray]:
    """Compute mean/std/min/max/slope along axis 0 (time).

    arr_t_first: shape (T, ...). Returns dict of arrays each shaped like
    arr_t_first[0] (i.e. time dimension reduced).
    """
    t = arr_t_first.shape[0]
    x = np.linspace(0.0, 1.0, t, dtype=np.float32)  # normalized time index
    mean = arr_t_first.mean(axis=0)
    std = arr_t_first.std(axis=0)
    amin = arr_t_first.min(axis=0)
    amax = arr_t_first.max(axis=0)
    # slope via least-squares on flattened spatial dims (vectorized polyfit)
    flat = arr_t_first.reshape(t, -1)
    x_c = x - x.mean()
    slope = (x_c[:, None] * (flat - flat.mean(axis=0, keepdims=True))).sum(axis=0) / (
        (x_c**2).sum() + EPS
    )
    slope = slope.reshape(arr_t_first.shape[1:])
    return {"mean": mean, "std": std, "min": amin, "max": amax, "slope": slope}


def patch_pixel_features(config: dict, patch_id: int) -> tuple[np.ndarray, list[str]]:
    """Per-pixel temporal-statistic feature cube for one patch.

    Returns (features, feature_names) where features has shape (n_features, H, W).
    Features: for each of the 10 bands and NDVI/NDWI, the temporal
    mean/std/min/max/slope -> 12 * 5 = 60 features per pixel.
    """
    s2 = to_reflectance(load_s2(config, patch_id))  # (T, C, H, W)
    ndvi = compute_ndvi(s2)  # (T, H, W)
    ndwi = compute_ndwi(s2)  # (T, H, W)

    feature_layers, names = [], []
    n_bands = s2.shape[1]
    for b in range(n_bands):
        band_name = config["bands"][b]["band"]
        stats = _temporal_stats(s2[:, b])
        for stat_name, arr in stats.items():
            feature_layers.append(arr)
            names.append(f"{band_name}_{stat_name}")
    for idx_name, idx_arr in (("ndvi", ndvi), ("ndwi", ndwi)):
        stats = _temporal_stats(idx_arr)
        for stat_name, arr in stats.items():
            feature_layers.append(arr)
            names.append(f"{idx_name}_{stat_name}")

    features = np.stack(feature_layers, axis=0)  # (n_features, H, W)
    return features, names


def patch_ndvi_series(config: dict, patch_id: int) -> tuple[np.ndarray, list[str]]:
    """Per-pixel *ordered* NDVI time series for one patch (NDVI-only track).

    Returns (features, feature_names) with features shaped (T, H, W) -- one
    feature per acquisition date, in chronological order, so feature i is the
    NDVI observed at acquisition i.

    Contrast with `patch_pixel_features`: four of that function's five
    statistics (mean/std/min/max) are permutation-invariant in time, so a
    model built on them can see *how much* vegetation there was but almost
    nothing about *when*. Keeping the raw ordered series lets a tree split on
    "NDVI at date 23 > 0.6", which encodes phenological timing directly --
    the signal that distinguishes autumn-sown cereals from summer row crops.
    The trade-off is that this track discards all non-NDVI spectral
    information (SWIR, red-edge), so it is a deliberately narrower feature set.

    Reads the precomputed NDVI cube from `cache.ndvi_dir` when available
    (see `src/build_ndvi_cache.py`), since deriving NDVI otherwise requires
    reading all 10 bands to use 2. The cache stores exactly the float32 array
    `compute_ndvi` returns, so results are bit-identical either way.
    """
    ndvi = None
    if config.get("cache", {}).get("use_ndvi_cache", False):
        cache_path = ndvi_cache_path(config, patch_id)
        if cache_path.exists():
            ndvi = np.load(cache_path)
    if ndvi is None:
        s2 = to_reflectance(load_s2(config, patch_id))  # (T, C, H, W)
        ndvi = compute_ndvi(s2)                          # (T, H, W)

    names = [f"ndvi_t{t:02d}" for t in range(ndvi.shape[0])]
    return ndvi, names


def ndvi_cache_path(config: dict, patch_id: int) -> Path:
    return PROJECT_ROOT / config["cache"]["ndvi_dir"] / f"NDVI_{patch_id}.npy"


def build_pixel_feature_table(
    config: dict,
    patch_ids: Iterable[int],
    seed: int | None = None,
    feature_fn=patch_pixel_features,
) -> pd.DataFrame:
    """Build a stratified-sampled per-pixel feature table for the ML track.

    Sampling strategy (documented per assignment task A): with 128*128=16384
    pixels per patch and up to ~100 patches, the full per-pixel table would
    be ~1.6M+ rows, which is unnecessary for a tree-based model and slow to
    iterate on. Instead we cap the number of pixels drawn *per class, per
    patch* at `preprocessing.max_pixels_per_patch_per_class` (config,
    default 400), sampled uniformly at random with a fixed seed. This keeps
    every patch and every class represented (important given regional/
    class imbalance) while bounding total rows and naturally downsampling
    the dominant classes (e.g. background, meadow) relative to rare ones.
    The void label (class 19, undefined/no data) is excluded entirely since
    it has no meaningful crop-type signal to learn.

    `feature_fn` selects which per-pixel representation to build:
    `patch_pixel_features` (temporal statistics, 60 features) or
    `patch_ndvi_series` (ordered NDVI series, T features). Sampling is
    identical either way, so tracks built on them are directly comparable.
    """
    rng = np.random.default_rng(seed if seed is not None else config["split"]["seed"])
    void_class = config["void_class"]
    cap = config["preprocessing"]["max_pixels_per_patch_per_class"]

    rows = []
    feature_names = None
    for pid in patch_ids:
        features, names = feature_fn(config, pid)  # (F, H, W)
        if feature_names is None:
            feature_names = names
        target = load_target(config, pid)  # (H, W)

        for cls in np.unique(target):
            if cls == void_class:
                continue
            ys, xs = np.where(target == cls)
            if len(ys) > cap:
                sel = rng.choice(len(ys), size=cap, replace=False)
                ys, xs = ys[sel], xs[sel]
            feat_vals = features[:, ys, xs].T  # (n_sel, F)
            df = pd.DataFrame(feat_vals, columns=feature_names)
            df["label"] = cls
            df["patch_id"] = pid
            df["row"] = ys
            df["col"] = xs
            rows.append(df)

    return pd.concat(rows, ignore_index=True)


def compute_band_normalization_stats(config: dict, patch_ids: Iterable[int]) -> dict:
    """Per-band mean/std (in reflectance units) computed over training patches only.

    Used to z-normalize inputs to the DL model; fitting on train-split patches
    only avoids leaking val/test statistics into preprocessing.
    """
    sums = None
    sq_sums = None
    count = 0
    for pid in patch_ids:
        s2 = to_reflectance(load_s2(config, pid))  # (T, C, H, W)
        # Move channel axis to last before flattening, since C is axis 1 here
        # (a plain .reshape(-1, C) would silently interleave bands/pixels).
        flat = np.moveaxis(s2, 1, -1).reshape(-1, s2.shape[1])  # (T*H*W, C)
        if sums is None:
            sums = flat.sum(axis=0, dtype=np.float64)
            sq_sums = (flat.astype(np.float64) ** 2).sum(axis=0)
        else:
            sums += flat.sum(axis=0, dtype=np.float64)
            sq_sums += (flat.astype(np.float64) ** 2).sum(axis=0)
        count += flat.shape[0]

    mean = sums / count
    var = sq_sums / count - mean**2
    std = np.sqrt(np.clip(var, EPS, None))
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}


if __name__ == "__main__":
    from data_loading import load_config, list_patch_ids

    cfg = load_config()
    ids = list_patch_ids(cfg)[:3]
    table = build_pixel_feature_table(cfg, ids)
    print(table.shape)
    print(table["label"].value_counts())
    stats = compute_band_normalization_stats(cfg, ids)
    print("band mean:", stats["mean"])
    print("band std:", stats["std"])
