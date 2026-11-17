#!/usr/bin/env python3
"""
longitudinal.py
===============

Two views for multi-hour / multi-day photometry with FED3 feeding. Pure
numpy / pandas (no Streamlit), shared by the group view and the tests.

1. Double-plotted actogram of the signal
   Each row is one calendar day; the row shows that day AND the next (0-48 h),
   so activity that crosses midnight is never cut in two. Colour = the binned
   signal; ticks = pellets. Double-plotted actograms are the standard way to
   show daily rhythms in circadian work, and long-term photometry papers show
   their signal the same way (e.g. Douglass et al. 2024, Cell Metabolism:
   recordings of up to a month, double-plotted heatmaps of the signal and of
   feeding).

2. Meal size vs response
   For every meal (FED3 pellets grouped by the meal definition) the change of
   the signal from just before the meal to just after it, against the size of
   the meal. Hunger-sensing neurons are inhibited in proportion to the
   calories eaten (AgRP: Beutler et al. 2017, Neuron; Su et al. 2017, Cell
   Reports), so a slope that differs from 0 means the response tracks how much
   was eaten, not just that eating started. Statistics treat the animal as the
   unit (slope per animal, then a one-sample test of the slopes; Loewinger
   et al. 2024, eLife), because meals from one animal are not independent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import fed3


# =============================================================================
# 1. ACTOGRAM
# =============================================================================

def actogram(t, y, clock_start=None, bin_s=300.0, pellets=None):
    """
    Binned signal by day and time of day.

    t, y         seconds on the recording's own axis, signal
    clock_start  local wall-clock time of t = 0, or None (then 'day' = 24 h since the start)
    pellets      pellet times (s, same axis), or None

    Returns dict:
      day_labels   list of str, one per row
      x            bin centres in hours, 0..48 (double-plotted)
      M            rows x 2*bins_per_day (row d = day d then day d + 1; NaN where no data)
      ticks_x, ticks_row   pellet ticks (hours 0..48, row index), each pellet drawn twice
      has_clock    bool
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ok = np.isfinite(t) & np.isfinite(y)
    t, y = t[ok], y[ok]
    nb = int(round(86400.0 / bin_s))
    bin_s = 86400.0 / nb
    if clock_start is not None:
        cs = pd.Timestamp(clock_start)
        midnight = cs.normalize()
        off = (cs - midnight).total_seconds()
    else:
        midnight, off = None, 0.0
    a = t + off                                  # seconds since midnight of day 0
    if a.size == 0:
        return None
    d_lo, d_hi = int(np.floor(a.min() / 86400.0)), int(np.floor(a.max() / 86400.0))
    nd = d_hi - d_lo + 1
    idx = np.floor((a - d_lo * 86400.0) / bin_s).astype(np.int64)
    flat = np.full(nd * nb, np.nan)
    s = np.bincount(idx, weights=y, minlength=nd * nb)
    c = np.bincount(idx, minlength=nd * nb)
    with np.errstate(invalid="ignore", divide="ignore"):
        flat = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    days = flat[: nd * nb].reshape(nd, nb)
    nxt = np.vstack([days[1:], np.full((1, nb), np.nan)])
    M = np.hstack([days, nxt])
    if midnight is not None:
        labels = [(midnight + pd.Timedelta(days=d_lo + i)).strftime("%a %d %b") for i in range(nd)]
    else:
        labels = [f"Day {i + 1}" for i in range(nd)]
    tx, tr = [], []
    if pellets is not None and len(pellets):
        p = np.asarray(pellets, float) + off
        p = p[(p >= a.min() - 1) & (p <= a.max() + 1)]
        pd_ = np.floor(p / 86400.0).astype(int) - d_lo
        ph = (p - (pd_ + d_lo) * 86400.0) / 3600.0
        for row, h in zip(pd_, ph):
            if 0 <= row < nd:
                tx.append(h)
                tr.append(row)
            if 0 <= row - 1 < nd:
                tx.append(h + 24.0)
                tr.append(row - 1)
    return {"day_labels": labels, "x": (np.arange(2 * nb) + 0.5) * bin_s / 3600.0, "M": M,
            "ticks_x": np.asarray(tx), "ticks_row": np.asarray(tr, int), "has_clock": midnight is not None,
            "bin_s": bin_s}


def dark_spans_48h(lights_on_h, lights_off_h):
    """Dark-phase spans (hours) on the 0-48 h axis of a double-plotted actogram."""
    if lights_on_h is None or lights_off_h is None or lights_on_h == lights_off_h:
        return []
    out = []
    for base in (0.0, 24.0):
        if lights_off_h < lights_on_h:            # e.g. dark 10:00 -> 22:00
            out.append((base + lights_off_h, base + lights_on_h))
        else:                                     # e.g. dark 19:00 -> 07:00 next day
            out.append((base + lights_off_h, base + 24.0 + lights_on_h))
            if base == 0.0:
                out.append((0.0, lights_on_h))
    return [(max(0.0, a), min(48.0, b)) for a, b in out if min(48.0, b) > max(0.0, a)]


# =============================================================================
# 2. MEAL SIZE VS RESPONSE
# =============================================================================

def _window_mean(t, y, a, b):
    m = (t >= a) & (t < b) & np.isfinite(y)
    return float(np.mean(y[m])) if m.sum() >= 2 else np.nan


def meal_responses(m, max_gap_s=60.0, min_pellets=1, base=(-300.0, -30.0), resp=(0.0, 600.0),
                   lights_on_h=None, lights_off_h=None):
    """
    One row per meal of a group member (dict from group_analysis.make_member, with its
    normalised signal in m['y']): onset, size, response = mean(resp window) - mean(base
    window), both relative to the meal's first pellet. Meals whose windows are not
    fully inside the recording are dropped.
    """
    cols = ["recording", "animal", "group", "onset_s", "pellets", "grams", "duration_s", "base", "after",
            "response", "hour_of_day", "dark"]
    if m.get("t") is None or not len(m.get("pellets", [])):
        return pd.DataFrame(columns=cols)
    t, y = m["t"], m["y"]
    meals = fed3.find_meals(m["pellets"], max_gap_s, min_pellets)
    rows = []
    for _, r in meals.iterrows():
        on = float(r["onset"])
        if on + base[0] < t[0] or on + resp[1] > t[-1]:
            continue
        b = _window_mean(t, y, on + base[0], on + base[1])
        a = _window_mean(t, y, on + resp[0], on + resp[1])
        hod, dark = np.nan, None
        if m.get("cs") is not None:
            ts = pd.Timestamp(m["cs"]) + pd.Timedelta(seconds=on)
            hod = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
            if lights_on_h is not None and lights_off_h is not None:
                dark = bool((lights_off_h <= hod < lights_on_h) if lights_off_h < lights_on_h
                            else (hod >= lights_off_h or hod < lights_on_h))
        rows.append({"recording": m["name"], "animal": m["animal"] or m["name"], "group": m["group"],
                     "onset_s": on, "pellets": int(r["n_pellets"]), "grams": int(r["n_pellets"]) * m["grams"],
                     "duration_s": float(r["duration_s"]), "base": b, "after": a, "response": a - b,
                     "hour_of_day": hod, "dark": dark})
    return pd.DataFrame(rows, columns=cols)


def slope_stats(df):
    """
    Per animal: n meals, Spearman rho and least-squares slope of response on meal size
    (pellets). Group rows: mean slope +/- SEM across animals and a one-sample t-test of
    the slopes against 0 (the animal is the unit). Returns (per_animal, per_group).
    """
    from scipy import stats as ss
    per = []
    for (g, an), sub in df.groupby(["group", "animal"], sort=False):
        sub = sub[np.isfinite(sub["response"])]
        row = {"group": g, "animal": an, "meals": int(len(sub)), "rho": np.nan, "p (rho)": np.nan,
               "slope per pellet": np.nan, "mean response": float(sub["response"].mean()) if len(sub) else np.nan}
        if len(sub) >= 3 and sub["pellets"].nunique() >= 2:
            rho, p = ss.spearmanr(sub["pellets"], sub["response"])
            sl = np.polyfit(sub["pellets"].to_numpy(float), sub["response"].to_numpy(float), 1)[0]
            row.update({"rho": float(rho), "p (rho)": float(p), "slope per pellet": float(sl)})
        per.append(row)
    per = pd.DataFrame(per)
    grp = []
    if not per.empty:
        for g, sub in per.groupby("group", sort=False):
            v = sub["slope per pellet"].to_numpy(float)
            v = v[np.isfinite(v)]
            r = {"group": g, "animals with a slope": int(v.size),
                 "mean slope per pellet": float(v.mean()) if v.size else np.nan,
                 "SEM": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else np.nan,
                 "t-test p (slope = 0)": np.nan}
            if v.size >= 3:
                r["t-test p (slope = 0)"] = float(ss.ttest_1samp(v, 0.0).pvalue)
            grp.append(r)
    return per, pd.DataFrame(grp)


__all__ = ["actogram", "dark_spans_48h", "meal_responses", "slope_stats"]
