"""Precompute and cache the per-patch NDVI cube.

NDVI uses 2 of the 10 Sentinel-2 bands, but deriving it requires reading the
whole `(T, 10, H, W)` int16 array (~15MB/patch, ~60MB once cast to float32).
Every NDVI experiment repeated that read, which dominated their runtime --
the Track D feature build spent ~13s/patch almost entirely on I/O. Caching
the `(T, H, W)` float32 NDVI cube (~3MB/patch) removes that cost from every
subsequent run.

The cache stores exactly the array `compute_ndvi()` returns, so cached and
on-the-fly runs produce bit-identical features (verified by `--verify`).

Usage:
    python build_ndvi_cache.py            # build the cache
    python build_ndvi_cache.py --verify   # rebuild-free check vs. on-the-fly
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from data_loading import PROJECT_ROOT, list_patch_ids, load_config, load_s2
from preprocessing import compute_ndvi, ndvi_cache_path, to_reflectance


def build_cache(config: dict, overwrite: bool = False) -> None:
    patch_ids = list_patch_ids(config)
    out_dir = PROJECT_ROOT / config["cache"]["ndvi_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    written = skipped = 0
    for i, pid in enumerate(patch_ids, 1):
        path = ndvi_cache_path(config, pid)
        if path.exists() and not overwrite:
            skipped += 1
            continue
        ndvi = compute_ndvi(to_reflectance(load_s2(config, pid)))  # (T, H, W) float32
        np.save(path, ndvi)
        written += 1
        if i % 20 == 0 or i == len(patch_ids):
            print(f"  {i}/{len(patch_ids)} patches ({time.time()-t0:.0f}s elapsed)")

    total_mb = sum(p.stat().st_size for p in out_dir.glob("NDVI_*.npy")) / 1e6
    print(f"\nwrote {written} patches, skipped {skipped} existing, in {time.time()-t0:.0f}s")
    print(f"cache: {out_dir}  ({total_mb:.0f} MB for {len(patch_ids)} patches)")


def verify(config: dict, n: int = 5) -> bool:
    """Confirm cached NDVI is bit-identical to computing it from raw bands."""
    patch_ids = list_patch_ids(config)[:n]
    ok = True
    for pid in patch_ids:
        path = ndvi_cache_path(config, pid)
        if not path.exists():
            print(f"  patch {pid}: NO CACHE FILE")
            ok = False
            continue
        cached = np.load(path)
        fresh = compute_ndvi(to_reflectance(load_s2(config, pid)))
        identical = cached.shape == fresh.shape and np.array_equal(cached, fresh)
        print(f"  patch {pid}: shape {cached.shape} dtype {cached.dtype} "
              f"bit-identical={identical}")
        ok &= identical
    print("\nVERIFY:", "cache matches on-the-fly computation exactly"
          if ok else "MISMATCH -- do not use the cache")
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="check cache against fresh computation")
    parser.add_argument("--overwrite", action="store_true", help="rebuild existing cache files")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.verify:
        raise SystemExit(0 if verify(cfg) else 1)
    build_cache(cfg, overwrite=args.overwrite)
    print()
    verify(cfg)
