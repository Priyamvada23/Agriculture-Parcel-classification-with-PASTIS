"""Harmonic decomposition and phenological metrics for NDVI time series.

WHY NOT CLASSICAL DECOMPOSITION
-------------------------------
Seasonal-trend decomposition (STL, `seasonal_decompose`) separates a series
into trend + seasonal + residual. It needs at least two full periods to make
that separation identifiable, and the standard implementations assume evenly
spaced samples. This dataset provides 46 acquisitions spanning 400 days --
**1.10 seasonal cycles** -- at irregular 5-25 day intervals. A "trend"
estimated from 1.1 cycles is indistinguishable from the seasonal term itself,
so STL would produce numbers that look meaningful and are not.

For the same reason, BFAST-style change detection does not apply: it exists to
locate abrupt land-cover change *between* years of monitoring, and there is
only one season here.

WHAT IS VALID ON A SINGLE IRREGULAR SEASON
------------------------------------------
1. **Harmonic (Fourier) decomposition.** Fitting sinusoids of known annual
   period by least squares handles irregular sampling natively (the design
   matrix is built from the real acquisition dates, not from sample index) and
   needs only one cycle. It splits the series into an interpretable seasonal
   component -- mean level, amplitude, and *phase*, i.e. when the peak occurs --
   plus a residual. Phase is exactly the discriminative signal: it is what
   separates autumn-sown cereals from summer row crops.

2. **Iterative outlier rejection (HANTS).** Cloud depresses NDVI, so
   contamination is one-sided. Fitting, discarding strongly negative
   residuals, and refitting reconstructs a cloud-free seasonal curve. This is
   the standard remote-sensing treatment (Harmonic ANalysis of Time Series)
   and doubles as principled cloud-date detection, replacing the crude
   "dataset-wide dip" heuristic used elsewhere in the EDA.

3. **Phenological metrics.** Start/peak/end of season, amplitude and seasonal
   integral, extracted from the reconstructed curve -- the standard TIMESAT
   descriptors of a crop calendar.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

PERIOD_DAYS = 365.25


def _design_matrix(days: np.ndarray, n_harmonics: int) -> np.ndarray:
    """Columns: [1, cos(2pi k t/T), sin(2pi k t/T) for k = 1..n_harmonics]."""
    cols = [np.ones_like(days, dtype=np.float64)]
    for k in range(1, n_harmonics + 1):
        w = 2.0 * np.pi * k * days / PERIOD_DAYS
        cols.append(np.cos(w))
        cols.append(np.sin(w))
    return np.column_stack(cols)


def harmonic_fit(
    values: np.ndarray,
    days: np.ndarray,
    n_harmonics: int = 2,
    reject_outliers: bool = True,
    max_iter: int = 10,
    reject_sigma: float = 2.5,
    max_reject_fraction: float = 0.25,
) -> dict:
    """Least-squares harmonic fit with optional one-sided outlier rejection.

    values : NDVI observations, shape (T,)
    days   : days elapsed from the first acquisition, shape (T,) -- real dates,
             so irregular spacing is handled correctly.

    Returns amplitude/phase of the annual harmonic, the reconstructed curve,
    R^2, and a boolean mask of points kept (False = rejected as cloud).

    On `reject_sigma`: each iteration recomputes sigma from the surviving
    residuals, so an aggressive threshold shrinks sigma and rejects more, which
    shrinks sigma again. At 1.5 this runaway discards >50% of some series and
    reports a flattering R^2 measured only on the survivors. 2.5 is the point
    at which recovered amplitude and peak date stabilise on this dataset;
    `max_reject_fraction` is a hard backstop against the same runaway.

    NOTE: `r2` is computed on retained points only. With rejection enabled it
    measures fit quality on the cloud-screened series, not on all observations.
    """
    values = np.asarray(values, dtype=np.float64)
    days = np.asarray(days, dtype=np.float64)
    keep = np.ones(len(values), dtype=bool)

    for _ in range(max_iter if reject_outliers else 1):
        A = _design_matrix(days[keep], n_harmonics)
        coef, *_ = np.linalg.lstsq(A, values[keep], rcond=None)
        fitted_all = _design_matrix(days, n_harmonics) @ coef
        resid = values - fitted_all
        if not reject_outliers:
            break
        # Cloud only ever depresses NDVI, so reject on the negative side only.
        sigma = resid[keep].std()
        if sigma < 1e-9:
            break
        # Intersecting with `keep` makes rejection monotonic: once a date is
        # discarded it stays discarded, so the loop always converges.
        new_keep = keep & (resid > -reject_sigma * sigma)
        min_keep = max(2 * n_harmonics + 2,
                       int(np.ceil(len(values) * (1.0 - max_reject_fraction))))
        if new_keep.sum() < min_keep or (new_keep == keep).all():
            break
        keep = new_keep

    A_all = _design_matrix(days, n_harmonics)
    fitted = A_all @ coef
    ss_res = ((values[keep] - fitted[keep]) ** 2).sum()
    ss_tot = ((values[keep] - values[keep].mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Annual harmonic (k=1): coef = [a0, a1, b1, a2, b2, ...]
    a1, b1 = coef[1], coef[2]
    amplitude = float(np.hypot(a1, b1))
    # Peak of a1*cos(wt) + b1*sin(wt) occurs at wt = atan2(b1, a1)
    phase_rad = float(np.arctan2(b1, a1))
    peak_day = float((phase_rad % (2 * np.pi)) / (2 * np.pi) * PERIOD_DAYS)

    return {
        "coef": coef,
        "fitted": fitted,
        "keep": keep,
        "n_rejected": int((~keep).sum()),
        "r2": float(r2),
        "mean_level": float(coef[0]),
        "amplitude": amplitude,
        "phase_rad": phase_rad,
        "peak_day_offset": peak_day,
        "residual": values - fitted,
    }


def phenology_metrics(
    values: np.ndarray,
    days: np.ndarray,
    dates: list[dt.date] | None = None,
    threshold: float = 0.5,
    use_harmonic: bool = True,
) -> dict:
    """TIMESAT-style descriptors of one seasonal NDVI curve.

    Computed on the harmonic reconstruction by default, so cloud drop-outs do
    not create spurious season starts/ends.

    Returns peak value/date, amplitude, start/end of season (crossings of
    `threshold` of the amplitude on the rising/falling limb), season length,
    and the seasonal integral (a proxy for cumulative greenness).
    """
    values = np.asarray(values, dtype=np.float64)
    days = np.asarray(days, dtype=np.float64)

    curve = harmonic_fit(values, days)["fitted"] if use_harmonic else values

    i_peak = int(np.argmax(curve))
    peak_value = float(curve[i_peak])

    # The 400-day window spans 1.1 seasons, so it contains the tail of the
    # PREVIOUS season as well as the current one. Searching for the start of
    # season from index 0 therefore finds the previous crop's senescence and
    # reports a nonsensical ~365-day season. Anchor the search on the local
    # minima bracketing the peak instead: start-of-season is the rise out of
    # the trough immediately preceding the peak, end-of-season the fall into
    # the trough immediately after it.
    i_trough_before = int(np.argmin(curve[: i_peak + 1])) if i_peak > 0 else 0
    i_trough_after = i_peak + int(np.argmin(curve[i_peak:])) if i_peak < len(curve) - 1 else len(curve) - 1

    base = float(min(curve[i_trough_before], curve[i_trough_after]))
    amplitude = peak_value - base
    level = base + threshold * amplitude

    def crossing(lo: int, hi: int, rising: bool) -> float | None:
        """Linear-interpolated day at which `curve` crosses `level`."""
        rng = range(lo, hi) if rising else range(hi, lo, -1)
        for i in rng:
            j = i + 1 if rising else i - 1
            if (curve[i] - level) * (curve[j] - level) <= 0 and curve[i] != curve[j]:
                f = (level - curve[i]) / (curve[j] - curve[i])
                return float(days[i] + f * (days[j] - days[i]))
        return None

    sos = crossing(i_trough_before, i_peak, rising=True) if i_peak > i_trough_before else None
    eos = crossing(i_peak, i_trough_after, rising=False) if i_trough_after > i_peak else None

    out = {
        "peak_value": peak_value,
        "peak_day": float(days[i_peak]),
        "base_value": base,
        "amplitude": float(amplitude),
        "sos_day": sos,
        "eos_day": eos,
        "season_length_days": (eos - sos) if (sos is not None and eos is not None) else None,
        "integral": float(np.trapezoid(curve - base, days)),
    }
    if dates is not None:
        start = dates[0]
        for key, day in (("peak_date", out["peak_day"]), ("sos_date", sos), ("eos_date", eos)):
            out[key] = (start + dt.timedelta(days=float(day))) if day is not None else None
    return out


def days_from_dates(dates: list[dt.date]) -> np.ndarray:
    """Days elapsed since the first acquisition -- the x-axis for every fit here."""
    d0 = dates[0]
    return np.array([(d - d0).days for d in dates], dtype=np.float64)
