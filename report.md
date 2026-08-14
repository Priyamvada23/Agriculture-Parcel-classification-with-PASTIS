# Crop-Type Classification on a PASTIS Sentinel-2 Subset

*Solafune Geo Data Scientist assignment — report.md*

---

## 1. Executive Summary

**Delivered model: a Random Forest on the per-pixel, chronologically-ordered
NDVI time series.** Each pixel is represented by its 46 NDVI values — one per
Sentinel-2 acquisition, in date order — and classified into one of 18
crop/land-cover classes.

**Evaluation protocol — crop pixels only.** The task is crop-type
classification, so metrics are computed over the 174,890 test pixels whose
ground truth is an actual crop (classes 1–18). Void (19) has no ground truth,
and background (0) is non-crop land cover; neither is a crop-classification
outcome, so neither is scored. Background is still a **training** class — the
model must be able to output it, or it would be forced to assign a crop to
roads, buildings and woodland. Only the scoring changes. §9.3 reports
background behaviour separately, including the blind spot this protocol
introduces.

Test-set performance (18 held-out patches, official PASTIS fold):

| Metric | Crop pixels only | *(background retained, for reference)* |
|---|---|---|
| Overall accuracy | **0.861** | *0.826* |
| Macro F1 (all 18 crop classes) | **0.370** | *0.379* |
| Macro F1 (14 classes present in test) | **0.475** | *0.481* |
| Weighted F1 | **0.876** | *0.818* |
| Mean IoU | **0.445** | *0.418* |

All six crop classes holding >10K test pixels — 93.9% of the cropped area —
score F1 between **0.89 and 0.94**.

**The central finding of this study is that the feature representation
mattered roughly three times more than the choice of model.** Four
alternatives were built and evaluated under an identical protocol. Holding
the model fixed and changing only how the time series is represented moved
macro-F1 by **+0.128**; swapping the model while holding the representation
fixed moved it by at most **+0.038**, and in the neural case moved it
*backwards*. Section 8 documents this evidence, which is the justification
for the delivered model.

A single caveat is carried prominently rather than buried: macro-F1 is
suppressed by the dataset, not only by the model. Four of the 18 crop classes
have **zero pixels in the test fold** and are scored 0 by construction; six
more have under 2,000 test pixels. Section 10 decomposes this.

---

## 2. Introduction & Problem Statement

Crop-type mapping from satellite image time series supports crop monitoring,
yield estimation, and land-use analysis. Given Sentinel-2 time series and
per-pixel crop labels for a small patch subset, the task is to build a
reproducible classification pipeline, evaluate it honestly, and reason about
its limitations — explicitly *not* to chase maximum accuracy. This report
therefore prioritises a defensible representation, an evaluation protocol
without spatial leakage, and error analysis that distinguishes model failure
from dataset limitation.

---

## 3. Dataset Description

- **Source**: subset of [PASTIS](https://github.com/VSainteuf/pastis-benchmark).
- **102 patches**, 128×128 px at 10 m, single Sentinel-2 tile (`t31tfm`,
  Lambert-93 / EPSG:2154, mainland France).
- **10 spectral bands** (B2–B12), **46 acquisitions per patch**, uniformly,
  spanning 2018-09-20 → 2019-10-25 (~13 months, a full crop season plus
  margin).
- **Labels**: semantic only — **no per-parcel instance IDs were provided**, so
  the task is pixel-level classification, not parcel classification. 20 class
  IDs (0 = Background, 1–18 = crop/land-cover, 19 = Void/no-data).
- **Official 5-fold spatial CV assignment** (`Fold` field in
  `metadata.geojson`), used for the split (§6).

---

## 4. Data Loading & Preparation

`src/data_loading.py` reads the `.npy` arrays and `metadata.geojson`;
`src/preprocessing.py` derives all features.

- **Reflectance scaling**: raw `int16` values are Sentinel-2 L2A reflectance
  ×10000 (observed range [-105, 14924]; small negatives are normal
  atmospheric-correction artifacts near zero reflectance, not errors) —
  divided by 10000 before any computation.
- **NDVI**: `(NIR − Red) / (NIR + Red)` = `(B8 − B4) / (B8 + B4)`, computed per
  acquisition, giving a `(46, 128, 128)` cube per patch.
- **Sampling strategy** (assignment task A): the full pixel table across all
  patches would be ~1.67M rows. Training instead draws up to **400 pixels per
  class per patch** (stratified, fixed seed 42), yielding 167,477 rows. This
  bounds training cost while keeping every patch and class represented. Note
  that the cap is a *compute* device, not the class-balancing mechanism —
  `class_weight='balanced'` fully equalises whatever distribution it receives;
  the cap's real contribution is geographic diversity per unit compute (400
  pixels from each of 66 patches rather than 8,000 from one).
- **Evaluation never samples**: validation and test metrics are always computed
  over **every** labelled pixel of the held-out patches (271,814 test pixels),
  so reported numbers reflect true full-image behaviour.
- **Void handling**: class 19 is excluded from training and from every reported
  metric — it marks absent ground truth, not a land-cover type.
- **NDVI cache**: NDVI needs 2 of 10 bands, but deriving it requires reading the
  whole `(46,10,128,128)` array (~15 MB, ~60 MB as float32). `src/build_ndvi_cache.py`
  precomputes the NDVI cube once (~3 MB/patch, 308 MB total). This cut the
  feature-table build for 84 patches from **995.7s to 1.4s**. The cache stores
  exactly the array `compute_ndvi()` returns; all 102 patches were verified
  **bit-identical** to on-the-fly computation, so results are unaffected.

---

## 5. Exploratory Data Analysis

Full analysis: `notebooks/exploration_and_training.ipynb` (executed, outputs
saved). A second notebook, `notebooks/npy_profile_inspection.ipynb`, provides
a per-patch raw-array inspector.

**Class distribution** — severe imbalance (`outputs/figures/class_distribution.png`):
Background 30.3%, Meadow 17.8%, Soft winter wheat 11.9%, Corn 11.8% — together
~72% of labelled pixels. Void is 8.6%. Orchard, Beet, Potatoes, Grapevine and
Winter durum wheat are each <0.1% or absent. This is why **macro-F1, not
accuracy**, is the headline metric.

**Imagery & labels** — RGB composites (B4/B3/B2) show parcel boundaries aligning
well with label polygons, confirming good spatial registration
(`outputs/figures/sample_patches_rgb_labels.png`).

**NDVI phenology** (`outputs/figures/ndvi_phenology_by_class.png`) — the finding
the delivered model is built on. Winter cereals green up in autumn, peak in
spring, and senesce before summer; summer row crops (corn, soybean, sunflower)
stay flat through winter and peak in July–August. **Crop identity is encoded in
*when* things happen**, not only in how green they get.

**AOI** — 102 patches scattered non-contiguously across a ~33 × 36 km bounding
box within one tile (`outputs/figures/aoi_extent.png`), consistent with PASTIS's
spatial-diversity sampling rather than one contiguous region.

**Data quality**:
- Sequence length is uniform (46/patch) — no padding or masking needed, and all
  patches share one acquisition calendar (important for §7 and §11).
- Several dates show a **dataset-wide NDVI collapse across every class
  simultaneously** — cloud/haze, not phenology. Sharpest: obs 3 (2018-10-10),
  obs 6 (2018-11-04), obs 36 (2019-08-11), with mean NDVI ≈ 0.01–0.03 versus a
  typical 0.2–0.5+. Milder dips at obs 7, 13, 26, 32. On individual patches this
  drives NDVI to physically impossible values (winter barley reaches −0.55 on
  2019-08-11), confirming these are artifacts.
- Background share per patch: mean 30.3%, max 66.7%. Void: mean 8.6%, max 29.2%.
- Parcels per patch: 21–82 (median 48); on average 70.5% of a patch lies inside
  a labelled parcel.

### 5.1 Time-series decomposition, trend & change detection

**Classical seasonal-trend decomposition and change detection do not apply to
this dataset, and saying so is more useful than producing numbers that look
meaningful.**

| Method requirement | This dataset |
|---|---|
| STL / `seasonal_decompose`: ≥2 full periods to separate trend from seasonal | **1.10 cycles** (400 days) |
| STL: evenly spaced samples | **irregular**, 5–25 day gaps |
| BFAST change detection: multi-year series to locate abrupt land-cover change | **one season** |

With 1.1 cycles the trend component is not identifiable from the seasonal term
— they are confounded, so an STL "trend" here would reflect the window length,
not the crops. BFAST-family methods exist to find land-cover change *between*
years; with a single season there is no such change to detect.

**What is valid on one irregularly-sampled season** (`src/phenology.py`):

1. **Harmonic (Fourier) decomposition** — least-squares sinusoids of known
   annual period. Handles irregular sampling natively (the design matrix uses
   real acquisition dates, not sample index) and needs only one cycle. Splits
   each curve into mean level, amplitude, and **phase** — *when* the peak
   occurs — plus a residual.
2. **Iterative outlier rejection (HANTS)** — cloud only depresses NDVI, so
   fitting, discarding strongly negative residuals and refitting both
   reconstructs a cloud-free curve and *detects* the contaminated dates. This
   is the form of change/anomaly detection that genuinely applies here, and it
   independently recovers the same cloud dates as the dip heuristic above.
3. **Phenological metrics** — SOS / POS / EOS, amplitude, seasonal integral
   (standard TIMESAT descriptors).

**Result — a crop calendar recovered from the data alone.** Nine of the 13
classes with ≥2000 px are shown; the full set is in
`outputs/metrics/phenology_metrics.csv`, with the figure at
`outputs/figures/phenology_decomposition.png`.

| Class | Amplitude | SOS | Peak | EOS | Season (d) |
|---|---|---|---|---|---|
| Winter barley | 0.250 | 2019-02-01 | **2019-04-13** | 2019-06-09 | 128 |
| Winter rapeseed | 0.209 | 2019-01-30 | **2019-05-03** | 2019-06-24 | 145 |
| Soft winter wheat | 0.283 | 2019-02-20 | **2019-05-03** | 2019-06-27 | 127 |
| Winter triticale | 0.240 | 2019-02-21 | **2019-05-03** | 2019-06-24 | 123 |
| Meadow | 0.140 | 2019-02-13 | 2019-05-03 | 2019-08-22 | 189 |
| Spring barley | 0.189 | 2019-04-03 | **2019-06-02** | 2019-08-04 | 123 |
| Sunflower | 0.179 | 2019-05-24 | **2019-07-17** | 2019-09-08 | 106 |
| Soybeans | 0.167 | 2019-06-13 | **2019-08-11** | 2019-09-28 | 107 |
| Corn | 0.144 | 2019-06-21 | **2019-08-11** | 2019-10-01 | 103 |

The peak-of-season ordering is a textbook crop calendar obtained with no
agronomic input: winter barley (13 Apr) → winter cereals and rapeseed (3 May)
→ spring barley (2 Jun) → sunflower (17 Jul) → corn and soybean (11 Aug). That
is a **120-day spread in peak date**.

**This is the quantitative basis for the delivered model.** In the
amplitude-versus-peak-date plane, classes separate far better *horizontally*
(timing) than *vertically*: amplitude spans a narrow 0.08–0.28 and overlaps
heavily between the winter and summer groups, while peak date splits them into
two clusters four months apart. A representation preserving *when* can exploit
that axis; order-invariant statistics cannot see it at all — §8 measures the
cost at **+0.128 macro-F1**.

Two caveats stated rather than buried:

- `r2` is computed on **retained** points, so it measures fit on the
  cloud-screened series. Aggressive rejection inflates it: at
  `reject_sigma=1.5` some classes lost >50% of observations and reported
  R²>0.97 on the survivors. The default 2.5 is where recovered amplitude and
  peak date stabilise, with a hard 25% cap on rejection.
- Low-amplitude, poorly-fit classes (Sorghum: amplitude 0.10, R²=0.42) have
  unreliable SOS/EOS — too flat for a threshold crossing to be well defined.
  Peak date stays meaningful; season length does not.

---

## 6. Train / Validation / Test Split

| Split | Folds | Patches |
|---|---|---|
| Train | 1, 2, 3 | 66 |
| Validation | 4 | 18 |
| Test | 5 | 18 |

The **official PASTIS fold assignment** is used rather than a random shuffle.
PASTIS's folds are spatially disjoint; a random per-patch split risks spatial
autocorrelation leakage, where neighbouring patches sharing soil, weather and
farm management land on both sides of the split and inflate reported
performance. Using the official folds also keeps results comparable to the
wider PASTIS literature. Patch IDs are written to
`configs/splits/{train,val,test}.txt` by `src/make_splits.py` so the split is
exactly reproducible.

This choice makes the numbers *lower* and more honest — that is its purpose.

---

## 7. Methodology — The Delivered Model

**Representation** (`src/preprocessing.py:patch_ndvi_series`): each pixel is a
46-dimensional vector `ndvi_t00 … ndvi_t45`, the NDVI at each acquisition in
chronological order. No summary statistics, no other bands.

The rationale is direct: a tree splitting on `ndvi_t28 > 0.6` asserts "still
green in mid-June," which is exactly the phenological statement that separates
autumn-sown cereals from summer row crops (§5). Preserving order preserves that
information.

**Model**: Random Forest — 300 trees, `max_depth=20`, `class_weight='balanced'`,
seed 42 (`configs/config.yaml: model_ndvi`).

**Class imbalance**: `class_weight='balanced'` (inverse-frequency), plus the
stratified sampling of §4.

**Compute**: CPU only. Feature build 1.4s with the NDVI cache (29s cold), Random
Forest fit 138s. The complete pipeline reproduces in **under three minutes** on
a laptop with no GPU.

**Reproduce**: `cd src && python build_ndvi_cache.py && python train.py --track ndvi`

---

## 8. Why This Representation — The Supporting Evidence

The delivered model was chosen on evidence, not preference. Three alternatives
were built and evaluated under the *identical* split, sampling, void handling
and metric code, so differences are attributable to the design change alone.

All figures below are crop-only (the primary protocol); the with-background
values are in `outputs/metrics/*_withbg_summary.json`.

| Track | Representation | Model | Accuracy | Macro F1 |
|---|---|---|---|---|
| **C (delivered)** | **46 ordered NDVI values** | **Random Forest** | **0.861** | **0.370** |
| D | 46 ordered NDVI values | TempCNN (1D CNN, 249K params) | 0.824 | 0.360 |
| A | 60 temporal statistics | XGBoost | 0.726 | 0.280 |
| A | 60 temporal statistics | Random Forest | 0.654 | 0.242 |
| B | raw `(T,C,H,W)`, all 10 bands | Temporal-Attention U-Net (491K params, 40 epochs) | 0.712 | 0.340 |

Isolating each variable:

| Comparison | Held fixed | Δ Macro F1 |
|---|---|---|
| temporal stats → ordered NDVI | model = Random Forest | **+0.128** |
| Random Forest → XGBoost | features = temporal stats | +0.038 |
| Random Forest → TempCNN | features = ordered NDVI | **−0.009** |

The representation effect is ~3× the largest model effect. Note that the
Random-Forest-over-TempCNN margin is small (−0.009 macro-F1, though the RF
also leads by 0.037 accuracy and 0.105 mean IoU); the defensible claim is that
the neural model brought **no benefit at this data scale**, not that it is
decisively worse.

**Why the statistics representation fails.** Track A described each band by its
temporal mean, std, min, max and slope. Four of those five are
**permutation-invariant in time** — shuffling the 46 acquisitions leaves them
bit-identical (verified directly; only `slope` changes). That representation
encodes *how much* vegetation there was and almost nothing about *when*. Given
that crop identity is a timing signal, this is a fundamental mismatch, and it
costs 0.128 macro-F1. Switching to the ordered series improved **every one of
the 14 crop classes present in the test set** (mean +0.164, no class made
worse), led by Sunflower (0.141 → 0.802) and Winter barley (0.383 → 0.920).

**Why the neural models did not win.** Both networks underperformed the Random
Forest on this data, and the reason is statistical rather than architectural:
**the effective sample size is 66 patches, not 167,477 pixels.** Pixels within a
parcel share soil, weather and one farmer's decisions — they are near-duplicates,
not independent observations. Fitting 249K–491K parameters against ~66
independent units overfits, while a Random Forest's bagging and feature
subsampling are robust in exactly that regime. The evidence is visible in
training: the TempCNN drove training loss from 0.87 to 0.10 while validation
macro-F1 plateaued at ~0.42 from epoch 15 onward.

The U-Net was additionally trained to full convergence (40 epochs, ~1.9 h CPU) to
ensure it was not being dismissed on an under-trained result. It improved from
0.293 to 0.332 macro-F1 — real, but insufficient, and still 0.228 behind on
accuracy. Its validation macro-F1 also oscillated between 0.219 and 0.400 (σ =
0.047) across the final 20 epochs, so its best-checkpoint value of 0.400 is
substantially selection-on-noise; test came in at 0.332. The delivered model
shows no comparable instability.

---

## 9. Results

### 9.1 Per-class performance (test set)

`outputs/metrics/ndvi_rf_test_per_class.csv`

| Class | Support (px) | Precision | Recall | F1 |
|---|---|---|---|---|
| Meadow | 54,359 | 0.939 | 0.853 | 0.894 |
| Soft winter wheat | 35,388 | 0.921 | 0.934 | 0.928 |
| Corn | 31,525 | 0.917 | 0.931 | 0.924 |
| Winter rapeseed | 17,328 | 0.948 | 0.924 | 0.936 |
| Soybeans | 13,828 | 0.913 | 0.924 | 0.919 |
| Winter barley | 11,780 | 0.939 | 0.901 | 0.920 |
| Sunflower | 2,851 | 0.943 | 0.698 | 0.802 |
| Fruits/vegetables/flowers | 2,643 | 0.000 | 0.000 | 0.000 |
| Leguminous fodder | 1,778 | 0.653 | 0.182 | 0.285 |
| Winter triticale | 1,612 | 0.067 | 0.035 | 0.046 |
| Mixed cereal | 674 | 0.000 | 0.000 | 0.000 |
| Sorghum | 665 | 0.000 | 0.000 | 0.000 |
| Winter durum wheat | 306 | 0.000 | 0.000 | 0.000 |
| Spring barley | 153 | 0.000 | 0.000 | 0.000 |
| Beet, Grapevine, Potatoes, Orchard | 0 | — | — | — |

Confusion matrix: `outputs/figures/confusion_matrix_ndvi_rf_test.png`.
The with-background version of this table is in
`outputs/metrics/ndvi_rf_test_withbg_per_class.csv`.

### 9.2 The model recovered the crop calendar

Because every feature *is* a date, the Random Forest's feature importances read
out directly as phenology (`outputs/metrics/ndvi_rf_feature_importance.csv`):

| Rank | Date | Importance |
|---|---|---|
| 1 | 2019-06-17 | 0.049 |
| 2 | 2019-06-02 | 0.048 |
| 3 | 2019-05-23 | 0.038 |
| 4 | 2019-08-21 | 0.035 |
| 5 | 2019-05-13 | 0.033 |

**Ten dates in the April–June window carry 30.5% of total importance** — the
period when winter cereals senesce while summer row crops are greening up, i.e.
precisely when the two groups are maximally separable. The model independently
rediscovered the agronomic discrimination window visible in the §5 NDVI curves,
which is meaningful evidence it learned real phenology rather than an artifact.
This interpretability is a practical argument for the representation
independent of its accuracy.

### 9.3 Background — the afterthought, quantified

Background is excluded from the headline metrics (§1) because it is not a crop.
That exclusion is deliberate but not free, and this section states its cost
rather than leaving it implicit.

| Direction | Count | Scored? |
|---|---|---|
| Crop pixels predicted as background | 11,701 / 174,890 (**6.7%**) | **Yes** — counts as a recall miss for that crop |
| Background pixels predicted as a crop | 23,024 / 96,924 (**23.8%**) | **No** — invisible under the crop-only protocol |

The second row is the blind spot. Roughly a quarter of background pixels are
assigned some crop label, and **79% of those errors land on Meadow**
(18,233 px), with the remainder spread over Corn (1,570), Soft winter wheat
(1,078), Winter barley (893) and Winter rapeseed (622).

The practical consequence is visible in the precision column: excluding
background-truth pixels removes those false positives from the denominator, so
crop precision rises — Meadow from **0.686 → 0.939**, Winter barley
0.870 → 0.939, Corn 0.874 → 0.917. Meadow is the extreme case because it is
spectrally and phenologically closest to unmanaged grassy background.

**How to read this.** For the assignment's question — *given a crop pixel, is
the crop identified correctly?* — the crop-only numbers are the right ones. For
an operational crop-mapping product, where the model would run over whole
scenes including non-agricultural land, the with-background numbers
(`*_withbg_*`) are the honest ones, because commission onto background is a
real map error. Both are reported for exactly this reason. A parcel mask, or the
parcel polygons already present in `metadata.geojson`, would remove most of this
error class in deployment (§12).

---

## 10. Analysis & Interpretation

### 10.1 Macro-F1 is suppressed by the dataset, not only the model

Reading 0.370 as "the model is 37% good" would be wrong. Decomposing it:

| Bucket | Classes | Share of crop px | Mean F1 |
|---|---|---|---|
| >10K px | 6 | **93.9%** | **0.920** |
| 2K–10K px | 2 | 3.1% | 0.401 |
| <2K px | 6 | 3.0% | 0.055 |
| **0 px** | **4** | **0%** | **0.000 (forced)** |

`Grapevine`, `Beet`, `Potatoes` and `Orchard` have **no pixels in the test
fold** — Beet appears nowhere in the subset at all, and the other three occur in
exactly one training patch each. They cannot be scored, yet they enter the macro
average as four hard zeros. Excluding them raises macro-F1 from 0.370 to
**0.475**. Both numbers are reported throughout rather than silently choosing
the flattering one.

The top row is the practical headline: on the six crop classes that make up
94% of the cropped area, mean F1 is **0.920**.

### 10.2 Rare-class failure is about patches, not pixels

| Class | Train patches | Train px | Test F1 |
|---|---|---|---|
| Meadow / Corn / Wheat | 60–65 | 115K–198K | 0.89–0.93 |
| Winter barley | 42 | 34,871 | 0.920 |
| Winter rapeseed | 38 | 56,030 | 0.936 |
| Winter triticale | 18 | 10,335 | 0.046 |
| Sunflower | 8 | 4,607 | 0.802 |
| Fruits/veg/flowers | 3 | 500 | 0.000 |
| Grapevine, Durum wheat, Potatoes, Orchard | **1** | 75–1,345 | n/a |

Every class in ≥38 training patches scores >0.89. Classes appearing in ≤3
patches fail entirely. Three patches of "Fruits/vegetables/flowers" is
effectively *n = 3*, regardless of pixel count — which is the same
patch-level-independence argument that explains the neural models' overfitting
(§8).

### 10.3 Confusions are agronomically coherent

Errors are systematic, not random — each failing class collapses into its
nearest agronomic relative:

- **Winter triticale → 75% Soft winter wheat.** Triticale is a wheat×rye hybrid
  with a nearly identical calendar; wheat has ~11× more training pixels, so the
  classifier has every incentive to absorb it. Notably, this class did *not*
  improve when timing was restored ,its failure is genuine
  spectral-temporal near-identity plus imbalance, not representation.
- **Sorghum → 81% Corn** (both summer C4 row crops, near-identical calendars).
- **Spring barley → Winter barley + wheat** (same genus).
- **Mixed cereal → scattered** across soybean/barley/wheat — it is by definition
  a mixture with no single signature.

That the errors are interpretable is evidence the features carry real signal;
the model is failing where the underlying classes are genuinely close, not at
random.

### 10.4 Effects of resolution, cloud, and timing

At 10 m, pixels near parcel boundaries mix multiple crops, producing the
residual salt-and-pepper noise visible in §11 — this is the main cost of
per-pixel classification with no spatial smoothing. The cloud-contaminated dates
identified in §5 inject noise directly into the feature vector: for this model,
a clouded acquisition corrupts one feature outright rather than being diluted
across a temporal average. That the model still performs well suggests the
Random Forest's redundancy across 46 correlated dates absorbs a few corrupted
ones. No acquisitions are missing (uniform 46/patch), which simplifies matters
relative to general PASTIS.

### 10.5 Limitations of the delivered model

- **Calendar dependence (most important).** Feature *i* means "the *i*-th
  acquisition." This is only comparable across patches because all 102 share one
  tile and one identical 46-date calendar. Applied to another tile, another year,
  or full PASTIS (where sequence length varies), the representation breaks unless
  the series is first resampled onto a fixed day-of-year grid. Track A's
  statistics, for all their weakness, are calendar-agnostic and would transfer
  unchanged. This is a real deployment trade-off.
- **Spectral information is unused.** The model discards SWIR and red-edge
  entirely. It won anyway, but that is unexploited signal.
- **Single split, not 5-fold CV.** PASTIS's protocol is 5-fold; one 3/1/1
  configuration was run. With 18 test patches, rare-class metrics have wide
  error bars and should be read as directional.
- **No spatial context.** Each pixel is classified independently, though parcel
  polygons are available in `metadata.geojson` and would support cheap
  post-processing (§12).
- **Confounded ablation.** Track C changes *both* ordering and band set relative
  to Track A, so "+0.128" credits ordering and NDVI-only jointly, not ordering
  alone.

---

## 11. Visualisation of Outputs

`outputs/figures/prediction_panel_patch_*.png` show Sentinel-2 RGB | ground
truth | predictions from all models on the same test patches, generated by
`src/visualization.py`.

The delivered model reproduces parcel geometry closely, with boundaries and
class assignments largely matching the ground truth. Residual error is
concentrated at parcel edges (mixed pixels) and as light within-parcel speckle —
expected, since each pixel is classified independently with no spatial
smoothing. For contrast, the temporal-statistics models show heavy
within-parcel noise and systematic class confusion across whole fields, and the
U-Net produces spatially smooth output that is often confidently assigned to the
wrong class.

---

## 12. Conclusion & Next Steps

A Random Forest on the ordered NDVI time series classifies crop types at **0.861
accuracy / 0.370 macro-F1 (0.475 over classes actually present)** on a
spatially-disjoint, leakage-free split, reproducible in under three minutes on a
CPU. It beat a gradient-boosted variant, a 1D CNN on identical features, and a
converged Temporal-Attention U-Net on raw bands. The evidence supports a
specific, transferable conclusion: **for crop-type classification on small
satellite time-series datasets, preserving phenological timing in the feature
representation mattered about three times more than model capacity** — and
model capacity actively hurt once the independent-sample count was this low.

Recommended next steps, in priority order:

1. **Combine ordering with the full spectrum** — ordered series of all 10 bands,
   or ordered NDVI concatenated with Track A statistics. This is the most
   promising cheap experiment and would disentangle "ordering" from "NDVI-only,"
   which the current comparison confounds.
2. **Make the representation calendar-robust** by resampling onto a fixed
   day-of-year grid (e.g. 24 fortnightly composites) before training, removing
   the dependence noted in §10.5 and enabling transfer to other tiles and years.
3. **Run all five folds** to replace single-split point estimates with means and
   variances, which the rare-class metrics badly need.
4. **Parcel-level majority voting** using the polygons already in
   `metadata.geojson` — cheap post-processing that should remove the residual
   within-parcel speckle.
5. **Explicit cloud handling** — drop or interpolate the dates flagged in §5,
   which corrupt individual features outright in this representation.
6. **Evaluate on full PASTIS** (~2,433 patches) to obtain statistically
   meaningful rare-class metrics, or group the rarest classes under a shared
   "other" label if data remains this scarce.

---

## 13. References

- Garnot, V. S. F. et al. "Panoptic Segmentation of Satellite Image Time Series
  with Convolutional Temporal Attention Networks." ICCV 2021. (PASTIS dataset /
  U-TAE architecture) — https://github.com/VSainteuf/pastis-benchmark
- Pelletier, C., Webb, G. I., Petitjean, F. "Temporal Convolutional Neural
  Network for the Classification of Satellite Image Time Series." Remote
  Sensing, 2019. (TempCNN, the Track D architecture family.)
- Rouse, J. W. et al. "Monitoring Vegetation Systems in the Great Plains with
  ERTS." NASA, 1974. (NDVI.)
- scikit-learn, XGBoost, PyTorch, GeoPandas, Rasterio — see `requirements.txt`.
