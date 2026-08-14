# PASTIS Crop-Type Classification

Solafune Geo Data Scientist take-home assignment: exploring a 102-patch
subset of the [PASTIS](https://github.com/VSainteuf/pastis-benchmark)
dataset (multi-temporal Sentinel-2 + semantic crop-type labels) and building
a crop classification pipeline. Full narrative writeup is in
[`report.md`](report.md); this README covers setup and reproduction.

## 1. Project overview

Given Sentinel-2 time series and per-pixel semantic crop-type labels for 102
128x128 patches (10m resolution, France), the goal is to classify each
labeled pixel into one of 18 crop/land-cover classes.

**Delivered model: a Random Forest on the per-pixel, chronologically ordered
NDVI time series** (46 features — one per acquisition date). Test-set
performance on a spatially disjoint split: **0.861 accuracy, 0.370 macro-F1**
(0.475 over the 14 classes actually present), **0.876 weighted-F1**,
**0.445 mIoU**. Reproduces in under three minutes on CPU.

**Metrics are computed over crop pixels only** (classes 1–18). Background (0)
is non-crop land cover and void (19) has no ground truth, so neither is a
crop-classification outcome. Background remains a *training* class — the model
must be able to output it. A reference set of metrics that retains background
is written alongside every result as `outputs/metrics/*_withbg_*`; see
`report.md` §9.3 for the trade-off this involves.

Three alternatives were built and evaluated under an identical protocol to
justify the choice (crop-only figures):

| Track | Representation | Model | Accuracy | Macro-F1 |
|---|---|---|---|---|
| **C (delivered)** | 46 ordered NDVI values | Random Forest | **0.861** | **0.370** |
| D | 46 ordered NDVI values | TempCNN (1D CNN) | 0.824 | 0.360 |
| A | 60 temporal statistics | XGBoost / Random Forest | 0.726 / 0.654 | 0.280 / 0.242 |
| B | raw `(T,C,H,W)`, 10 bands | Temporal-Attention U-Net (40 ep) | 0.712 | 0.340 |

**Key finding: the feature representation mattered ~3x more than the model.**
Changing only the representation (model held fixed) moved macro-F1 by
**+0.128**; changing only the model (representation held fixed) moved it by at
most +0.038, and *negatively* for the neural net. Ordered NDVI preserves
phenological *timing*, which the temporal statistics discard — 4 of their 5
statistics are permutation-invariant in time.

Harmonic decomposition of the NDVI series (`src/phenology.py`, `report.md`
§5.1) shows why: peak-of-season recovers a textbook crop calendar with no
agronomic input — winter barley 13 Apr → winter cereals 3 May → sunflower
17 Jul → corn and soybean 11 Aug, a **120-day spread**. Classes separate along
that timing axis far more cleanly than by amplitude. Classical STL
decomposition and BFAST change detection are *not* applicable here (the series
spans 1.10 seasonal cycles at irregular 5–25 day intervals); §5.1 explains
what is valid instead.

See [`report.md`](report.md) (or [`report.tex`](report.tex) for the LaTeX
version) for full results and discussion.

## 2. Dataset structure

This repo does **not** include the PASTIS data itself (too large for
GitHub). Place the provided subset at the project root as:

```
project-root/
└── PASTIS_subset/
    ├── DATA_S2/
    │   └── S2_<patch_id>.npy          # (46, 10, 128, 128) int16, reflectance x10000
    ├── ANNOTATIONS/
    │   └── TARGET_<patch_id>.npy      # (1, 128, 128) uint8, semantic class per pixel
    └── metadata.geojson                # per-patch fold, dates, tile, parcel stats
```

`configs/config.yaml` points at this path (`data.root`); update it if you
place the data elsewhere.

## 3. Environment setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Requires Python 3.10+. `torch` is CPU-only by default via PyPI; for GPU
training (recommended for the DL track — see §6) install the CUDA build
per [pytorch.org](https://pytorch.org/get-started/locally/) instead, or run
on Google Colab where a CUDA build is preinstalled.

## 4. Project structure

```
project-root/
├── README.md
├── report.md                  # main report
├── report.tex                 # same report as editable LaTeX (pdflatex report.tex, twice)
├── requirements.txt
├── notebooks/
│   ├── exploration_and_training.ipynb   # EDA (tasks A/B) + pipeline orchestration,
│   │                                    #   fully executed with saved outputs
│   └── npy_profile_inspection.ipynb     # per-patch raw-array inspector: shapes, dtypes,
│                                        #   band/NDVI profiles, per-class NDVI curves
├── src/
│   ├── data_loading.py        # .npy / metadata.geojson readers, dataset inspection
│   ├── preprocessing.py       # NDVI/NDWI, ordered NDVI series (delivered model),
│   │                          #   temporal-stat features (Track A), normalization
│   ├── phenology.py           # harmonic (Fourier) decomposition, HANTS cloud rejection,
│   │                          #   TIMESAT phenology metrics (SOS/POS/EOS) — report §5.1
│   ├── build_ndvi_cache.py    # precompute + cache NDVI cubes; --verify checks
│   │                          #   they are bit-identical to on-the-fly computation
│   ├── make_splits.py         # writes configs/splits/{train,val,test}.txt
│   ├── model_ml.py            # tree-model training + full-patch inference
│   ├── model_ndvi_dl.py       # TempCNN 1D architecture (Track D)
│   ├── model_dl.py            # Temporal-Attention U-Net architecture (Track B)
│   ├── train_dl.py            # U-Net training loop (dataset, augmentation, checkpoints)
│   ├── train.py               # CLI: --track ndvi | ml | ndvi_dl | dl
│   ├── evaluate.py            # shared metrics + evaluate_and_save(): the single place
│   │                          #   the scoring protocol is defined for every track
│   ├── recompute_metrics.py   # re-score saved predictions under a changed protocol,
│   │                          #   without retraining
│   ├── predict_split.py       # run a saved checkpoint over a split (inference only),
│   │                          #   to regenerate predictions without retraining
│   ├── build_comparison.py    # cross-model comparison CSV, derived from the metric
│   │                          #   files so it cannot drift from them
│   └── visualization.py       # RGB|GT|prediction panels, confusion-matrix heatmaps
├── outputs/
│   ├── figures/                # EDA + evaluation plots (tracked)
│   ├── metrics/                 # per-split metrics; *_withbg_* = background-retained
│   │                            #   reference protocol; phenology_metrics.csv (tracked)
│   ├── predictions/             # per-patch predicted label maps, .npz (tracked)
│   └── models/                  # trained weights — gitignored, see §7
├── cache/ndvi/                 # precomputed NDVI cubes — gitignored, regenerate with
│                               #   src/build_ndvi_cache.py (308 MB)
└── configs/
    ├── config.yaml               # single source of truth: paths, bands, classes,
    │                             #   hyperparams, evaluation protocol
    └── splits/{train,val,test}.txt
```

## 5. How to run

```bash
cd src

# 1. Reproducible train/val/test split (official PASTIS folds; run once)
python make_splits.py

# 2. Cache the NDVI cubes (run once; ~5 min, 308MB — makes all NDVI work ~700x faster)
python build_ndvi_cache.py

# --- the delivered model (CPU, under 3 minutes) ---
python train.py --track ndvi

# --- alternatives, for the comparison in report.md §8 ---
python train.py --track ml        # Track A: temporal statistics + RF/XGBoost (~10 min)
python train.py --track ndvi_dl   # Track D: 1D TempCNN on the NDVI series (~10 min)
python train.py --track dl        # Track B: Temporal-Attn U-Net (~1.9 h on CPU, see §6)

# Confusion matrices + RGB|GT|prediction panels for every trained model
python visualization.py

# Re-score every saved prediction set under a changed metric protocol,
# without retraining (edit evaluation.ignore_classes in config.yaml first)
python recompute_metrics.py

# Run saved checkpoints over a split without retraining, then rebuild the
# cross-model comparison table from the metric files
python predict_split.py --split val
python build_comparison.py --split test
```

Or open `notebooks/exploration_and_training.ipynb` for the narrated EDA
(already executed, with all figures saved to `outputs/figures/`), and
`notebooks/npy_profile_inspection.ipynb` to inspect any single patch's raw
arrays and NDVI profiles interactively.

The phenological decomposition in `report.md` §5.1 is produced by that first
notebook via `src/phenology.py`, writing
`outputs/metrics/phenology_metrics.csv` and
`outputs/figures/phenology_decomposition.png`.

All scripts read `configs/config.yaml` by default (`--config` to override).
Hyperparameters, band/class definitions, and the fold-to-split mapping all
live there — nothing is hardcoded in the scripts.

## 6. Deep learning track & compute constraints

The Temporal-Attention U-Net (`src/model_dl.py`) is deliberately small
(~490K parameters) given only 66 training patches. On this development
machine (no local GPU) one epoch over the 66 training patches takes
**~167s (2.8 min)**, so the full `model_dl.epochs: 40` configuration runs in
**~1.9 hours on CPU** — slow but perfectly feasible locally, and that is how
the reported DL results were produced. `src/train.py --track dl` auto-detects
CUDA and falls back to CPU with a warning if unavailable.

**Optional: run on Google Colab (free GPU tier) for a much faster turnaround**

```python
# In a Colab notebook:
from google.colab import drive
drive.mount('/content/drive')
# Upload/copy this repo and PASTIS_subset/ to Drive first, then:
%cd /content/drive/MyDrive/<path-to-repo>
!pip install -r requirements.txt
!cd src && python make_splits.py && python train.py --track dl
```

This produces `outputs/models/dl_model.pt`, `outputs/metrics/dl_*`, and
`outputs/predictions/dl_test_predictions.npz` identical in format to a local
run — copy them back into this repo's `outputs/` to regenerate
`report.md`'s figures/tables via `src/visualization.py`.

The DL results in `report.md` come from the **full 40-epoch CPU run**. The
earlier reduced 10-epoch run is retained under
`outputs/metrics/dl10ep_*` (and `outputs/models/dl10ep_model.pt`) so the
effect of training to convergence can be inspected directly.

## 7. Model weights & exact reproduction

All trained weights are written to `outputs/models/`. What ships in the repo
differs by model, for a specific reason:

| Checkpoint | Size | In repo? | Why |
|---|---|---|---|
| `ndvi_dl_model.pt` (TempCNN) | 1.0 MB | **yes** | small, and *not* verified bit-reproducible |
| `dl_model.pt` (U-Net) | 1.9 MB | **yes** | same |
| `dl_band_stats.npz` | <1 KB | **yes** | normalisation stats the U-Net needs at inference |
| `ndvi_rf_model.joblib` (**delivered**) | 90 MB | no | retrains bit-identically in ~3 min (below) |
| `rf_model.joblib` | 139 MB | no | as above |
| `xgb_model.joblib` | 17 MB | no | as above |

The tree models are excluded because even compressed they sit past GitHub's
50 MB warning threshold, and — unlike the neural ones — they are provably
reproducible, so shipping them buys nothing.

### The delivered model reproduces exactly

`train.py` is seeded from `configs/config.yaml` (`split.seed: 42`, plus
`random_state` on each estimator). This was verified rather than assumed:
retraining the delivered model from scratch and comparing against the saved
checkpoint gives

- **bit-identical predictions across all 294,912 test pixels** (18 patches), and
- **identical feature importances**.

So `python train.py --track ndvi` recreates *the* model, not merely an
equivalent one. It takes about 3 minutes on CPU once the NDVI cache is built.

```bash
cd src
python make_splits.py          # deterministic split from the official folds
python build_ndvi_cache.py     # ~5 min once; --verify asserts bit-identical NDVI
python train.py --track ndvi   # ~3 min -> outputs/models/ndvi_rf_model.joblib
```

**The neural tracks carry a caveat.** Their training loops are seeded
(`torch.manual_seed`, `np.random.seed`), but full bit-reproducibility was not
verified — that would need a 1.9-hour rerun for the U-Net — and PyTorch does
not guarantee identical results across platforms, thread counts or library
versions without `torch.use_deterministic_algorithms(True)`. Their weights are
therefore committed, which sidesteps the question entirely.

**To share the tree models anyway**, attach the compressed `.joblib` files to
a GitHub Release (2 GB per file, no LFS quota) or track them with Git LFS:

```bash
git lfs install && git lfs track "outputs/models/*.joblib"
git add .gitattributes outputs/models/*.joblib
```

`model_ml.save_model()` compresses at level 6 by default, which is what brings
the delivered forest from 775 MB to 90 MB; verified lossless (predictions
identical before and after the round-trip).

## 8. Key assumptions & limitations

- Only the semantic annotation layer (0th layer) was provided — no
  per-parcel instance IDs — so classification is pixel/patch-level, not
  parcel-level, throughout.
- Traditional-ML training uses a class-stratified pixel sample (up to 400
  px/class/patch) rather than every pixel, for tractability; validation and
  test metrics are always computed on the **full** pixel grid of held-out
  patches. See `preprocessing.build_pixel_feature_table`'s docstring.
- The void label (class 19) is excluded from training and from all reported
  metrics. Background (class 0) is **trained on but not scored**: it is
  non-crop land cover, so it is not a crop-classification outcome. Metrics
  that retain it are written to `outputs/metrics/*_withbg_*` — these are the
  more conservative numbers for an operational crop map, since crop-only
  scoring does not penalise predicting a crop on background land
  (23.8% of background pixels, mostly Meadow; see `report.md` §9.3).
- Train/val/test uses the official PASTIS `Fold` field (folds 1-3 / 4 / 5)
  rather than a random split, to avoid spatial-neighborhood leakage.
- Several classes (Orchard, Beet, Potatoes, Grapevine, Winter durum wheat)
  have near-zero or zero pixel support in this 102-patch subset — their
  per-class metrics are not statistically meaningful here; see `report.md`
  §10.1 for the decomposition of what macro-F1 is actually measuring.
- The delivered representation indexes features by **acquisition number**, so
  it assumes all patches share one acquisition calendar — true here (single
  tile, identical 46 dates) but not in general. Applying it to another tile or
  year requires resampling onto a fixed day-of-year grid first; `report.md`
  §10.5 covers this.
- Harmonic decomposition is used for the phenology analysis only, not for the
  delivered features. `phenology.harmonic_fit` reports R² on **retained**
  points, so it reflects fit to the cloud-screened series rather than to all
  46 observations.
- See `report.md` §10 for full interpretation: class confusions, effects of
  class imbalance, cloud-contaminated acquisitions, and limitations.
