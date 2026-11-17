#!/usr/bin/env python3
"""
baseline.py
===========

Normalising a recording to a BASELINE PERIOD you choose instead of the whole
recording, and measuring a response against that baseline. Pure numpy, no
Streamlit, so the single-recording view, the group view and the tests share it.

Why
---
A z-score over the whole recording uses the mean and SD of everything,
responses included. A big, long response raises the SD and shifts the mean,
so the same peak gets a smaller z the more of it is inside the normalisation
window. Example: 5 min of flat baseline, then a peak that returns to baseline
3 min later. Over the whole 8 min the peak may reach z = 2; against the 5 min
of baseline alone (the period with no response) it may reach z = 8. Only the
baseline-only number answers "how far did the signal move from where it was
before the event?". This is the usual definition in event-related photometry
(baseline window before each event; e.g. Simpson et al. 2024, Neuron;
Martianova et al. 2019, JoVE).

Definitions (baseline = the `length` seconds that END at `end`)
--------------------------------------------------------------
  mean_b, sd_b   mean and SD of dF/F over the baseline
  z_b(t)         (dF/F(t) - mean_b) / sd_b
  % change(t)    (1 + dF/F(t)) / (1 + mean_b) - 1, i.e. the change of the
                 motion-corrected fluorescence relative to its baseline level
                 (dF/F is a fraction here; the app's columns are in percent)
"""

from __future__ import annotations

import numpy as np

MIN_BASELINE_SAMPLES = 5


def window_mask(t, end, length):
    t = np.asarray(t, float)
    return (t >= end - length) & (t < end)


def baseline_stats(t, y, end, length):
    """Mean, SD and size of y in the baseline [end - length, end)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    m = window_mask(t, end, length) & np.isfinite(y)
    n = int(m.sum())
    if n < MIN_BASELINE_SAMPLES:
        return {"ok": False, "n": n, "mean": np.nan, "sd": np.nan, "start": end - length, "end": end,
                "covered_sec": 0.0}
    yy = y[m]
    covered = float(t[m].max() - t[m].min()) if n > 1 else 0.0
    return {"ok": True, "n": n, "mean": float(yy.mean()), "sd": float(yy.std(ddof=1)), "start": end - length,
            "end": end, "covered_sec": covered}


def z_to_baseline(y, stats):
    y = np.asarray(y, float)
    if not stats.get("ok") or not np.isfinite(stats["sd"]) or stats["sd"] <= 0:
        return np.full_like(y, np.nan)
    return (y - stats["mean"]) / stats["sd"]


def pct_change_dff(dff_percent, mean_percent):
    """% change of the corrected fluorescence from the baseline, for dF/F given in percent."""
    d = np.asarray(dff_percent, float)
    return ((1.0 + d / 100.0) / (1.0 + mean_percent / 100.0) - 1.0) * 100.0


def smooth_by_time(t, y, sec):
    """Centred running mean over `sec` seconds (NaN-aware); unchanged when sec <= 0."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if sec is None or sec <= 0 or y.size < 3:
        return y.copy()
    dt = np.nanmedian(np.diff(t)) if t.size > 1 else 1.0
    k = int(max(1, round(sec / dt)))
    if k <= 1:
        return y.copy()
    ok = np.isfinite(y)
    ker = np.ones(k)
    s = np.convolve(np.where(ok, y, 0.0), ker, mode="same")
    c = np.convolve(ok.astype(float), ker, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = s / c
    out[c == 0] = np.nan
    return out


def peak_vs_baseline(t, dff_percent, end, length, response_sec, direction="auto", smooth_sec=1.0,
                     return_z=1.0, onset_z=2.0):
    """
    Measure the response after a baseline.

    t, dff_percent  the recording (seconds, dF/F in %)
    end, length     baseline = [end - length, end)
    response_sec    response window = [end, end + response_sec]
    direction       "auto" (larger excursion), "up" or "down"
    smooth_sec      running mean applied before finding the peak (noise would
                    otherwise make the single highest sample the peak)

    Returns a dict (all times relative to `end`, the end of the baseline):
      baseline_mean_pct, baseline_sd_pct, baseline_n
      direction, peak_time_s, peak_dff_pct (dF/F at the peak, as recorded),
      peak_change_pct (% change from the baseline level), peak_z (vs the baseline SD),
      mean_z (mean z over the response window), auc_z (z x seconds over the window),
      onset_s (first time |z| >= onset_z in the peak's direction),
      return_s (first time after the peak that |z| <= return_z, None if it never does),
      duration_s (return_s - onset_s),
      peak_z_whole_window (z of the same peak if the mean and SD were taken over baseline +
      response together, the "whole window" normalisation), whole_mean_pct, whole_sd_pct
      ok, message
    """
    t = np.asarray(t, float)
    y = np.asarray(dff_percent, float)
    out = {"ok": False, "message": ""}
    b = baseline_stats(t, y, end, length)
    out.update(baseline_mean_pct=b["mean"], baseline_sd_pct=b["sd"], baseline_n=b["n"])
    if not b["ok"]:
        out["message"] = "The baseline window holds too few samples (is it inside the recording?)."
        return out
    if not np.isfinite(b["sd"]) or b["sd"] <= 0:
        out["message"] = "The signal is flat during the baseline (SD = 0), so z cannot be computed."
        return out
    ys = smooth_by_time(t, y, smooth_sec)
    r = (t >= end) & (t <= end + response_sec) & np.isfinite(ys)
    if r.sum() < 3:
        out["message"] = "The response window holds too few samples (is it inside the recording?)."
        return out
    tr, yr = t[r], ys[r]
    z = (yr - b["mean"]) / b["sd"]
    if direction == "auto":
        direction = "up" if np.nanmax(z) >= -np.nanmin(z) else "down"
    i = int(np.nanargmax(z) if direction == "up" else np.nanargmin(z))
    sgn = 1.0 if direction == "up" else -1.0
    peak_v = float(yr[i])
    onset = None
    hit = np.flatnonzero(sgn * z[: i + 1] >= onset_z)
    if hit.size:
        onset = float(tr[hit[0]] - end)
    ret = None
    back = np.flatnonzero(np.abs(z[i:]) <= return_z)
    if back.size:
        ret = float(tr[i + back[0]] - end)
    dt = np.diff(tr)
    auc = float(np.nansum(0.5 * (z[1:] + z[:-1]) * dt)) if tr.size > 1 else 0.0
    whole = (t >= end - length) & (t <= end + response_sec) & np.isfinite(y)
    wm, ws = float(np.mean(y[whole])), float(np.std(y[whole], ddof=1))
    yw = ys[(t >= end - length) & (t <= end + response_sec)]
    out.update(
        ok=True, direction=direction, peak_time_s=float(tr[i] - end), peak_dff_pct=peak_v,
        peak_change_pct=float(pct_change_dff(peak_v, b["mean"])), peak_z=float(z[i]),
        mean_z=float(np.nanmean(z)), auc_z=auc, onset_s=onset, return_s=ret,
        duration_s=(ret - onset) if (ret is not None and onset is not None) else None,
        whole_mean_pct=wm, whole_sd_pct=ws,
        peak_z_whole_window=float((peak_v - wm) / ws) if ws > 0 else np.nan,
        response_end_s=float(min(response_sec, t[np.isfinite(y)].max() - end)),
        max_window_z=float(np.nanmax(np.abs((yw - wm) / ws))) if ws > 0 and yw.size else np.nan,
    )
    if ret is None:
        out["message"] = (f"The signal had not returned to within {return_z:g} SD of the baseline by the end of "
                          "the response window; lengthen the window to see the return.")
    return out


def describe(res, length):
    """One plain-language sentence for a peak_vs_baseline result."""
    if not res.get("ok"):
        return res.get("message", "")
    word = "peak" if res["direction"] == "up" else "dip"
    s = (f"The {word} came {res['peak_time_s']:.1f} s after the baseline ended: dF/F {res['peak_change_pct']:+.1f} % "
         f"from the baseline level, z = {res['peak_z']:+.2f} against the {length / 60:g}-min baseline alone. "
         f"Normalised over baseline + response together it would read z = {res['peak_z_whole_window']:+.2f}")
    s += (", because the response itself then inflates the SD and moves the mean." if
          abs(res["peak_z_whole_window"]) < abs(res["peak_z"]) else ".")
    if res.get("return_s") is not None:
        s += f" It was back within 1 SD of the baseline {res['return_s']:.1f} s after the baseline ended."
    return s


__all__ = ["window_mask", "baseline_stats", "z_to_baseline", "pct_change_dff", "smooth_by_time",
           "peak_vs_baseline", "describe"]
