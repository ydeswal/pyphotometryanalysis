#!/usr/bin/env python3
"""
group_analysis.py
=================

Combining several recordings (and their FED3 data) into group results.
Pure numpy / pandas, no Streamlit, so the group view, the AI chat and the
tests all use the same numbers.

Methods follow long-term photometry practice
--------------------------------------------
* Douglass et al. 2024, Cell Metabolism (pyPhotometry + Doric mini-cubes,
  recordings up to a month): long traces smoothed with a 30-min running mean
  and shown in 1-h bins; group data as mean +/- SEM across mice on a ZT axis;
  double-plotted heatmaps; meals = bouts separated by >= 300 s; meal-aligned
  traces baseline-subtracted.
* Loewinger et al. 2024, eLife: the animal, not the single event, is the
  unit for group statistics; pooling events from different animals inflates
  n (pseudo-replication).
* Matikainen-Ankney et al. 2021, eLife (FED3): meals = pellets < 1 min
  apart; intake as pellets per hour, mean +/- SEM across mice.

Units
-----
Times are seconds. Each member keeps its own clock; `align_offset` maps it
onto the shared x axis of the chosen alignment.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

import fed3
import interpret

ALIGNMENTS = ["Clock time (same moment)", "Time of day (overlay days)", "Time since recording start",
              "Time since first pellet", "Time since an event"]
NORMS = ["z-score (whole recording)", "Robust z (median / MAD)", "z-score to a baseline period",
         "Subtract baseline mean", "% change from baseline mean", "As recorded",
         "Scale 0–1 (1st–99th percentile)"]
STATS = ["Mean ± SEM", "Mean ± 95% CI (bootstrap)", "Median & IQR", "Mean ± SD"]
FED_MODES = ["Pellets per bin", "Cumulative intake (g)", "Cumulative, % of own total", "Animals eating (%)",
             "Eating rate (pellets/h, rolling)"]

METHOD_NOTES = {
    "z-score (whole recording)": "Each recording centred on its own mean and divided by its own SD, so animals with "
                                 "different expression levels or fibre placements can be averaged. It removes "
                                 "absolute size, so compare amplitudes between animals in dF/F, not z.",
    "Robust z (median / MAD)": "Like z-score but uses median and MAD (x1.4826), so long recordings with big "
                               "transients or artefacts do not inflate the SD. Good default for 12-48 h recordings.",
    "z-score to a baseline period": "Mean and SD taken from a baseline period only (first minutes of each recording, "
                                    "or before the first pellet). Shows change from that state.",
    "% change from baseline mean": "Each recording divided by its own baseline mean: 100 × (F − F_baseline) / "
                                   "F_baseline. The usual way to compare RAW channels (465, 405, 465 ÷ 405) between "
                                   "animals, because raw volts depend on LED power and fibre. Not for z-scores.",
    "Subtract baseline mean": "Keeps the original units (e.g. dF/F %) and removes each recording's baseline offset.",
    "As recorded": "No normalisation. Only meaningful for dF/F; raw volts differ between animals and set-ups.",
    "Scale 0–1 (1st–99th percentile)": "Each recording mapped to its own range. Useful to compare the timing of "
                                       "patterns, not their size.",
    "Mean ± SEM": "Standard for group photometry and intake plots (e.g. Douglass et al. 2024, FED3 2021).",
    "Mean ± 95% CI (bootstrap)": "Resamples animals with replacement (2,000 times). Makes no normality assumption; "
                                 "with 3-5 animals the interval is wide, which is honest.",
    "Median & IQR": "Robust to one unusual animal; better than the mean when n is small and one animal differs.",
    "Mean ± SD": "Shows the spread between animals rather than the precision of the mean.",
    "Pellets per bin": "Count per bin for each animal, then the group statistic. Bins where an animal's FED was not "
                       "logging are missing for that animal, not zero.",
    "Cumulative intake (g)": "Running total from the start of the window, then the group statistic. Missing FED time "
                             "adds nothing, so an animal with gaps will look lower.",
    "Cumulative, % of own total": "Each animal's running total as a percentage of its own total. Removes differences "
                                  "in how much each animal eats and shows WHEN they eat.",
    "Animals eating (%)": "Share of animals (with FED logging) that took at least one pellet in the bin. Robust with "
                          "small groups.",
    "Eating rate (pellets/h, rolling)": "Pellets per hour in a centred rolling window for each animal, then the group "
                                        "statistic. Smooth view of when the group eats.",
}


# =============================================================================
# MEMBERS
# =============================================================================

def make_member(name, animal, group, t, y, clock_start, fed=None, sessions=None, fit_r2=None,
                control=None, pellet_grams=0.02):
    """
    One recording (or one FED-only animal when t is None).

    t, y        signal, t in seconds on the recording's own axis
    clock_start Timestamp of t = 0 (local wall clock)
    control     (t, z) of the drift-removed 405 control, or None
    """
    m = {"name": str(name), "animal": animal, "group": str(group or "All"), "t": None, "y": None,
         "cs": pd.Timestamp(clock_start) if clock_start is not None else None, "pellets": np.array([]),
         "cov": [], "fit_r2": fit_r2, "control": control, "grams": float(pellet_grams), "events": [],
         "event_t": None}
    if t is not None:
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        o = np.argsort(t, kind="stable")
        m["t"], m["y"] = t[o], y[o]
    if fed is not None and animal and animal in fed.animals and m["cs"] is not None:
        m["pellets"] = np.sort(fed3.to_seconds(fed.times(animal, "pellet", sessions), m["cs"]))
        cov = fed.coverage_of(animal, sessions)
        m["cov"] = list(zip(fed3.to_seconds(cov["start"], m["cs"]), fed3.to_seconds(cov["stop"], m["cs"])))
    return m


def window_of(m):
    """[start, end] seconds of a member on its own axis."""
    if m["t"] is not None and m["t"].size:
        return float(m["t"][0]), float(m["t"][-1])
    if m["cov"]:
        return float(min(a for a, _ in m["cov"])), float(max(b for _, b in m["cov"]))
    if m["pellets"].size:
        return float(m["pellets"][0]), float(m["pellets"][-1])
    return 0.0, 0.0


def align_offset(m, alignment, ref_clock=None):
    """Seconds to ADD to the member's own times to put them on the shared axis,
    or None if this member cannot be aligned that way."""
    if alignment.startswith("Clock"):
        if m["cs"] is None or ref_clock is None:
            return None
        return (m["cs"] - pd.Timestamp(ref_clock)).total_seconds()
    if alignment.startswith("Time of day"):
        if m["cs"] is None:
            return None
        return (m["cs"] - m["cs"].normalize()).total_seconds()
    if alignment.startswith("Time since recording"):
        return -window_of(m)[0] if m["t"] is None else 0.0
    if alignment.startswith("Time since first pellet"):
        a, b = window_of(m)
        p = m["pellets"][(m["pellets"] >= a) & (m["pellets"] <= b)]
        return -float(p[0]) if p.size else None
    if alignment.startswith("Time since an event"):
        return -float(m["event_t"]) if m.get("event_t") is not None else None
    raise ValueError(alignment)


def reference_clock(members):
    cs = [m["cs"] for m in members if m["cs"] is not None]
    return min(cs).floor("min") if cs else None


# =============================================================================
# SIGNAL
# =============================================================================

def normalise(t, y, method, baseline_mask=None):
    """Normalise one recording. baseline_mask selects the baseline samples."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return y.copy()
    if method.startswith("As recorded"):
        return y.copy()
    if method.startswith("z-score (whole"):
        sd = np.nanstd(y)
        return (y - np.nanmean(y)) / sd if sd > 0 else y - np.nanmean(y)
    if method.startswith("Robust z"):
        med = np.nanmedian(y)
        mad = np.nanmedian(np.abs(y - med)) * 1.4826
        return (y - med) / mad if mad > 0 else y - med
    if method.startswith("Scale 0"):
        lo, hi = np.nanpercentile(y, [1, 99])
        return (y - lo) / (hi - lo) if hi > lo else y - lo
    b = ok if baseline_mask is None else (np.asarray(baseline_mask, bool) & ok)
    if b.sum() < 3:
        return np.full_like(y, np.nan)
    if method.startswith("z-score to a baseline"):
        sd = np.std(y[b])
        return (y - np.mean(y[b])) / sd if sd > 0 else y - np.mean(y[b])
    if method.startswith("Subtract baseline"):
        return y - np.mean(y[b])
    if method.startswith("% change"):
        mb = np.mean(y[b])
        return (y / mb - 1.0) * 100.0 if mb != 0 else np.full_like(y, np.nan)
    raise ValueError(method)


def baseline_mask(m, alignment, minutes):
    """Baseline samples: the first `minutes` of the recording, or the
    `minutes` before the first pellet / the event when aligning to it."""
    t = m["t"]
    if alignment.startswith(("Time since first pellet", "Time since an event")):
        off = align_offset(m, alignment)
        if off is None:
            return np.zeros(t.size, bool)
        x = t + off
        return (x >= -minutes * 60.0) & (x < 0)
    return t <= t[0] + minutes * 60.0


def bin_mean(x, y, e0, bin_s, n_bins):
    """Mean of y in regular bins starting at e0 (NaN where empty)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    idx = np.floor((x[ok] - e0) / bin_s).astype(np.int64)
    keep = (idx >= 0) & (idx < n_bins)
    idx, yy = idx[keep], y[ok][keep]
    s = np.bincount(idx, weights=yy, minlength=n_bins)
    c = np.bincount(idx, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out


def running_mean(v, k):
    """NaN-aware centred running mean over k bins (k <= 1: unchanged)."""
    v = np.asarray(v, float)
    if k <= 1 or v.size == 0:
        return v
    ok = np.isfinite(v)
    ker = np.ones(int(k))
    s = np.convolve(np.where(ok, v, 0.0), ker, mode="same")
    c = np.convolve(ok.astype(float), ker, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c < max(1, k / 3)] = np.nan
    out[~ok] = np.nan
    return out


def grid_for(members, alignment, bin_s, ref_clock=None):
    """Bin edges covering every member on the shared axis."""
    lo, hi = [], []
    for m in members:
        off = align_offset(m, alignment, ref_clock)
        if off is None:
            continue
        a, b = window_of(m)
        lo.append(a + off)
        hi.append(b + off)
    if not lo:
        return None
    e0 = math.floor(min(lo) / bin_s) * bin_s
    n = int(math.ceil((max(hi) - e0) / bin_s))
    return e0, max(n, 1)


def signal_matrix(members, alignment, norm, bin_s, smooth_bins=1, baseline_min=30.0, ref_clock=None,
                  per_animal=True):
    """
    Binned, normalised signal of every member on the shared axis.
    Returns dict: x (bin centres), rows (labels), M (rows x bins), groups,
    animals, skipped (names that could not be aligned).
    With per_animal=True recordings of the same animal in the same group are
    averaged first, so each animal counts once.
    """
    mem = [m for m in members if m["t"] is not None]
    g = grid_for(mem, alignment, bin_s, ref_clock)
    if g is None:
        return None
    e0, n = g
    rows, M, groups, animals, skipped = [], [], [], [], []
    for m in mem:
        off = align_offset(m, alignment, ref_clock)
        if off is None:
            skipped.append(m["name"])
            continue
        bm = baseline_mask(m, alignment, baseline_min) if "baseline" in norm.lower() else None
        y = normalise(m["t"], m["y"], norm, bm)
        v = running_mean(bin_mean(m["t"] + off, y, e0, bin_s, n), smooth_bins)
        rows.append(m["name"])
        M.append(v)
        groups.append(m["group"])
        animals.append(m["animal"] or m["name"])
    if not M:
        return None
    M = np.vstack(M)
    if per_animal:
        M, rows, groups, animals = _collapse(M, rows, groups, animals)
    return {"x": e0 + (np.arange(n) + 0.5) * bin_s, "e0": e0, "n_bins": n, "bin_s": bin_s, "rows": rows,
            "M": M, "groups": groups, "animals": animals, "skipped": skipped}


def _collapse(M, rows, groups, animals):
    """Average rows that share (group, animal)."""
    keys = list(dict.fromkeys(zip(groups, animals)))
    if len(keys) == len(rows):
        return M, rows, groups, animals
    out, r, g, a = [], [], [], []
    for gk, ak in keys:
        sel = [i for i in range(len(rows)) if groups[i] == gk and animals[i] == ak]
        with np.errstate(invalid="ignore"):
            out.append(np.nanmean(M[sel], axis=0) if len(sel) > 1 else M[sel[0]])
        r.append(f"{ak} (mean of {len(sel)})" if len(sel) > 1 else rows[sel[0]])
        g.append(gk)
        a.append(ak)
    return np.vstack(out), r, g, a


def group_stat(M, stat="Mean ± SEM", n_boot=2000, seed=0):
    """Centre line and band across rows (animals), NaN-aware.
    Returns centre, lo, hi, n (rows with data per bin)."""
    M = np.asarray(M, float)
    n = np.sum(np.isfinite(M), axis=0)
    import warnings
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if stat.startswith("Median"):
            c = np.nanmedian(M, axis=0)
            lo, hi = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
        else:
            c = np.nanmean(M, axis=0)
            sd = np.nanstd(M, axis=0, ddof=1)
            if stat.endswith("SD"):
                lo, hi = c - sd, c + sd
            elif "bootstrap" in stat:
                lo, hi = _boot_ci(M, n_boot, seed)
            else:
                se = sd / np.sqrt(n)
                lo, hi = c - se, c + se
    one = n < 2
    lo = np.where(one, np.nan, lo)
    hi = np.where(one, np.nan, hi)
    c = np.where(n == 0, np.nan, c)
    return c, lo, hi, n


def _boot_ci(M, n_boot, seed):
    import warnings
    rng = np.random.default_rng(seed)
    r = M.shape[0]
    means = []
    chunk = max(1, int(2e6 // max(1, M.size)))
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        idx = rng.integers(0, r, size=(k, r))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            means.append(np.nanmean(M[idx], axis=1))
        done += k
    means = np.vstack(means)
    return np.nanpercentile(means, 2.5, axis=0), np.nanpercentile(means, 97.5, axis=0)


# =============================================================================
# FED
# =============================================================================

def fed_matrix(members, alignment, bin_s, mode="Pellets per bin", ref_clock=None, rate_window_s=1800.0,
               per_animal=True):
    """
    FED intake of every member on the shared axis, limited to each member's
    window (its recording, or its FED logging for FED-only animals).
    Returns dict like signal_matrix, with M in the units of `mode`.
    """
    mem = [m for m in members if (m["pellets"].size or m["cov"])]
    g = grid_for(mem, alignment, bin_s, ref_clock)
    if g is None:
        return None
    e0, n = g
    edges = e0 + np.arange(n + 1) * bin_s
    rows, M, groups, animals, skipped = [], [], [], [], []
    for m in mem:
        off = align_offset(m, alignment, ref_clock)
        if off is None:
            skipped.append(m["name"])
            continue
        a, b = window_of(m)
        p = m["pellets"][(m["pellets"] >= a) & (m["pellets"] <= b)] + off
        cov = [(max(x, a) + off, min(y, b) + off) for x, y in (m["cov"] or [(a, b)]) if min(y, b) > max(x, a)]
        bins = fed3.intake_bins(p, e0, e0 + n * bin_s, bin_s, cov)
        cnt = bins["pellets"].to_numpy(float)[:n].copy()
        inside = (edges[1:] > a + off) & (edges[:-1] < b + off)
        cnt[~inside] = np.nan
        if mode.startswith("Pellets per bin"):
            v = cnt
        elif mode.startswith("Cumulative"):
            cum = np.nancumsum(np.where(np.isfinite(cnt), cnt, 0.0))
            v = np.where(inside, cum, np.nan)
            if "%" in mode:
                tot = np.nanmax(v) if np.isfinite(v).any() else 0
                v = v / tot * 100.0 if tot > 0 else v * np.nan
            else:
                v = v * m["grams"]
        elif mode.startswith("Animals eating"):
            v = np.where(np.isfinite(cnt), (cnt > 0).astype(float) * 100.0, np.nan)
        elif mode.startswith("Eating rate"):
            v = fed3.rolling_rate(p, e0 + (np.arange(n) + 0.5) * bin_s, rate_window_s)
            v = np.where(np.isfinite(cnt), v, np.nan)
        else:
            raise ValueError(mode)
        rows.append(m["name"])
        M.append(v)
        groups.append(m["group"])
        animals.append(m["animal"] or m["name"])
    if not M:
        return None
    M = np.vstack(M)
    if per_animal:
        M, rows, groups, animals = _collapse(M, rows, groups, animals)
    return {"x": e0 + (np.arange(n) + 0.5) * bin_s, "e0": e0, "n_bins": n, "bin_s": bin_s, "rows": rows,
            "M": M, "groups": groups, "animals": animals, "skipped": skipped}


def meal_table(members, max_gap_s=60.0, min_pellets=2, lights_on_h=None, lights_off_h=None):
    """
    Feeding pattern of every member inside its window, plus group rows
    (mean ± SEM per group). Measures follow the FED3 and meal-pattern
    literature: meals, meal size, duration, inter-meal interval, satiety
    ratio (minutes of not eating after a meal per pellet in it).
    """
    out = []
    for m in members:
        a, b = window_of(m)
        p = m["pellets"][(m["pellets"] >= a) & (m["pellets"] <= b)]
        cov = [(max(x, a), min(y, b)) for x, y in (m["cov"] or [(a, b)]) if min(y, b) > max(x, a)]
        hours = sum(y - x for x, y in cov) / 3600.0
        meals = fed3.find_meals(p, max_gap_s, min_pellets)
        nxt = np.r_[meals["onset"].to_numpy(float)[1:], np.nan] - meals["offset"].to_numpy(float) \
            if len(meals) else np.array([])
        sat = (nxt / 60.0) / meals["n_pellets"].to_numpy(float) if len(meals) else np.array([])
        row = {
            "group": m["group"], "animal": m["animal"] or "", "recording": m["name"],
            "hours logged": hours, "pellets": int(p.size), "grams": p.size * m["grams"],
            "pellets / h": p.size / hours if hours else np.nan,
            "meals": int(len(meals)),
            "meal size (pellets)": float(meals["n_pellets"].mean()) if len(meals) else np.nan,
            "meal duration (min)": float(meals["duration_s"].mean() / 60) if len(meals) else np.nan,
            "inter-meal interval (min)": float(np.nanmean(meals["gap_before_s"]) / 60)
            if len(meals) > 1 else np.nan,
            "satiety ratio (min/pellet)": float(np.nanmean(sat)) if np.isfinite(sat).any() else np.nan,
            "% pellets in meals": float(meals["n_pellets"].sum() / p.size * 100) if p.size else np.nan,
            "first pellet (min from start)": float((p[0] - a) / 60) if p.size else np.nan,
        }
        if lights_on_h is not None and lights_off_h is not None and m["cs"] is not None and p.size:
            hod = ((m["cs"].hour * 3600 + m["cs"].minute * 60 + m["cs"].second + p) / 3600.0) % 24
            if lights_on_h < lights_off_h:
                dark = (hod < lights_on_h) | (hod >= lights_off_h)
            else:
                dark = (hod >= lights_off_h) & (hod < lights_on_h)
            row["% pellets in dark"] = float(dark.mean() * 100)
        out.append(row)
    df = pd.DataFrame(out)
    if df.empty:
        return df, df
    num = [c for c in df.columns if c not in ("group", "animal", "recording")]
    summ = []
    for gname, sub in df.groupby("group", sort=False):
        k = len(sub)
        r = {"group": gname, "animals": k}
        for c in num:
            v = sub[c].to_numpy(float)
            v = v[np.isfinite(v)]
            r[c] = (f"{v.mean():.3g} ± {v.std(ddof=1) / np.sqrt(v.size):.2g}" if v.size > 1 else
                    (f"{v[0]:.3g}" if v.size else "n/a"))
        summ.append(r)
    return df, pd.DataFrame(summ)


# =============================================================================
# MEAL-ALIGNED, GROUP
# =============================================================================

def member_meal_response(m, max_gap_s=300.0, min_pellets=1, pre=300.0, post=600.0, base=(-100.0, -50.0),
                         resp=(0.0, 50.0), dt=None, n_perm=1000):
    """
    One animal: baseline-subtracted signal around meal onsets, its mean curve,
    the per-meal changes, a permutation test against random times, the 405
    control, and the fixed trust verdict.
    """
    if m["t"] is None:
        return None
    t, y = m["t"], m["y"]
    a, b = window_of(m)
    p = m["pellets"][(m["pellets"] >= a) & (m["pellets"] <= b)]
    on = fed3.find_meals(p, max_gap_s, min_pellets)["onset"].to_numpy(float)
    dt = dt or max(float(np.median(np.diff(t))), 0.25)
    mat, tax, kept = fed3.aligned_matrix(t, y, on, pre, post, baseline=base, dt=dt)
    cov = [(max(x, a), min(z, b)) for x, z in (m["cov"] or [(a, b)]) if min(z, b) > max(x, a)]
    st = fed3.permutation_test(t, y, kept, base, resp, allowed_sec=cov, n_perm=n_perm)
    deltas = fed3.window_means(t, y, kept, *resp) - fed3.window_means(t, y, kept, *base)
    ctrl = {"effect": None, "p": None}
    if m.get("control") is not None and kept.size >= 3:
        ct, cu = m["control"]
        c = fed3.permutation_test(ct, cu, kept, base, resp, allowed_sec=cov, n_perm=n_perm)
        if np.isfinite(c["p"]):
            ctrl = {"effect": float(c["observed"] - c["null_mean"]), "p": float(c["p"])}
    check = interpret.assess_response(st["n"], st["observed"], st["p"], st["null_mean"], ctrl["effect"],
                                      ctrl["p"], m.get("fit_r2"), unit="meals")
    curve = np.nanmean(mat, axis=0) if mat.shape[0] else np.full(tax.size, np.nan)
    return {"name": m["name"], "animal": m["animal"] or m["name"], "group": m["group"], "tax": tax,
            "curve": curve, "n_meals": int(kept.size), "meals_found": int(on.size),
            "deltas": deltas[np.isfinite(deltas)], "deltas_all": deltas, "mean_delta": float(np.nanmean(deltas)) if kept.size else np.nan,
            "p_vs_random": st["p"], "null_mean": st["null_mean"], "control": ctrl, "check": check,
            "meal_times": kept}


def group_meal_response(responses, stat="Mean ± SEM"):
    """
    Combine per-animal responses. Animals are the unit: each contributes its
    own mean curve and its own mean change. Returns per group: tax, centre,
    lo, hi, n_animals, animal-level tests, pooled-meal numbers (for
    reference only), and the group verdict.
    """
    from scipy import stats as ss
    res = [r for r in responses if r is not None and r["n_meals"] > 0]
    out = {}
    for g in dict.fromkeys(r["group"] for r in res):
        rr = [r for r in res if r["group"] == g]
        tax = rr[0]["tax"]
        M = np.vstack([np.interp(tax, r["tax"], r["curve"], left=np.nan, right=np.nan) for r in rr])
        c, lo, hi, n = group_stat(M, stat)
        d = np.array([r["mean_delta"] for r in rr], float)
        d = d[np.isfinite(d)]
        tt = ss.ttest_1samp(d, 0.0) if d.size >= 2 else None
        try:
            w = ss.wilcoxon(d) if d.size >= 2 and np.any(d != 0) else None
        except ValueError:
            w = None
        pooled = np.concatenate([r["deltas"] for r in rr]) if rr else np.array([])
        n_sig = sum(1 for r in rr if r["check"]["level"] in ("good", "moderate", "uncertain"))
        n_sig_clean = sum(1 for r in rr if r["check"]["level"] in ("good", "moderate"))
        p_animals = float(tt.pvalue) if tt is not None else np.nan
        check = _group_check(rr, d, p_animals)
        min_w = 2.0 / 2 ** d.size if d.size else np.nan
        out[g] = {"tax": tax, "centre": c, "lo": lo, "hi": hi, "n": n, "animals": [r["animal"] for r in rr],
                  "animal_deltas": d, "mean_delta": float(np.mean(d)) if d.size else np.nan,
                  "sem_delta": float(np.std(d, ddof=1) / np.sqrt(d.size)) if d.size > 1 else np.nan,
                  "t_p": p_animals, "t_stat": float(tt.statistic) if tt is not None else np.nan,
                  "wilcoxon_p": float(w.pvalue) if w is not None else np.nan, "wilcoxon_min_p": min_w,
                  "n_animals": int(d.size), "n_sig_animals": n_sig, "n_sig_clean": n_sig_clean, "pooled_meals": int(pooled.size),
                  "pooled_mean": float(np.mean(pooled)) if pooled.size else np.nan, "check": check}
    return out


def _group_check(rr, d, p_animals):
    """
    Fixed verdict for a group: the animal-level test, then the 405 control
    animal by animal. If the 405 control changes at meals in any animal, the
    group result is at most "uncertain", and the test is repeated on the
    animals whose 405 control is flat.
    """
    from scipy import stats as ss
    base = interpret.assess_response(d.size, float(np.mean(d)) if d.size else np.nan, p_animals, 0.0,
                                     None, None, None, unit="animals", min_n=4,
                                     versus="no change (mean of each animal's meals, t-test)")
    sig = base["level"] != "not significant" and base["level"] != "not testable"
    known = [r for r in rr if r["control"]["p"] is not None]
    moved = [r for r in known if r["control"]["p"] < interpret.ALPHA]
    clean = [r for r in known if r["control"]["p"] >= interpret.ALPHA and np.isfinite(r["mean_delta"])]
    first = base["verdict"].split(". ")[0].rstrip(".") + "."
    reasons = [x for x in base["reasons"] if "could not be checked" not in x]
    must = [x for x in base["must_say"] if "not checked" not in x]
    level = base["level"]
    if not known:
        if sig:
            level = "moderate"
            reasons.append("The 405 control could not be checked in any animal.")
            must.append("the 405 control was not checked")
        verdict = first
    elif moved:
        names = ", ".join(r["animal"] for r in moved)
        reasons.append(f"In {len(moved)} of {len(known)} animals ({names}) the 405 control itself changes "
                       "significantly at meals, so their 465 change may not be neural (movement, chewing, "
                       "blood flow, or over-correction by the 405 fit).")
        verdict = first
        if sig:
            level = "uncertain"
            verdict += (f" But the 405 control also changes at meals in {len(moved)} of {len(known)} animals, "
                        "so the data cannot show the response is neural in all of them.")
        must.append("the 405 control changed significantly in some animals, so the response may not be neural")
        dc = np.array([r["mean_delta"] for r in clean], float)
        if dc.size >= 2:
            tc = ss.ttest_1samp(dc, 0.0)
            reasons.append(f"Animals with a flat 405 control only ({', '.join(r['animal'] for r in clean)}): "
                           f"mean change {dc.mean():+.3g}, t-test p = {tc.pvalue:.2g} (n = {dc.size}).")
        elif dc.size == 1:
            reasons.append(f"Only one animal ({clean[0]['animal']}) has a flat 405 control.")
        else:
            reasons.append("No animal has a flat 405 control.")
    else:
        verdict = first
        if sig:
            verdict += f" The 405 control is flat at meals in all {len(known)} animals checked."
        reasons.append(f"The 405 control does not change at meals in any of the {len(known)} animals checked.")
    if sig and d.size < 4 and level == "good":
        level = "moderate"
    if d.size and d.size <= 6:
        reasons.append(f"Only {d.size} animals: a Wilcoxon test cannot reach p below {2.0 / 2 ** d.size:.3g}, "
                       "and one unusual animal moves the mean a lot.")
    return {"level": level, "label": interpret.LEVEL_TEXT[level], "verdict": verdict, "reasons": reasons,
            "must_say": must}


def compare_groups(responses, g1, g2):
    """Animal-level comparison of mean meal responses between two groups.
    Paired when the same animals are in both (e.g. day 1 vs day 7)."""
    from scipy import stats as ss
    a = {r["animal"]: r["mean_delta"] for r in responses if r and r["group"] == g1 and np.isfinite(r["mean_delta"])}
    b = {r["animal"]: r["mean_delta"] for r in responses if r and r["group"] == g2 and np.isfinite(r["mean_delta"])}
    both = [k for k in a if k in b]
    if len(both) >= 2 and len(both) >= 0.5 * min(len(a), len(b)):
        x = np.array([a[k] for k in both])
        y = np.array([b[k] for k in both])
        t = ss.ttest_rel(y, x)
        try:
            w = ss.wilcoxon(y - x)
            wp = float(w.pvalue)
        except ValueError:
            wp = np.nan
        return {"design": f"paired ({len(both)} animals in both)", "diff": float(np.mean(y - x)),
                "t_p": float(t.pvalue), "nonparam_p": wp, "nonparam": "Wilcoxon signed-rank"}
    x, y = np.array(list(a.values())), np.array(list(b.values()))
    if x.size < 2 or y.size < 2:
        return None
    t = ss.ttest_ind(y, x, equal_var=False)
    u = ss.mannwhitneyu(y, x)
    return {"design": f"independent ({x.size} vs {y.size} animals)", "diff": float(y.mean() - x.mean()),
            "t_p": float(t.pvalue), "nonparam_p": float(u.pvalue), "nonparam": "Mann-Whitney U"}
