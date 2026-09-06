#!/usr/bin/env python3
"""
lickometer.py
=============

Lickometer ingestion, bout detection, and peri-event photometry alignment.

Lick data can arrive three ways:
  1. A pyPhotometry digital input (digital_1 / digital_2) recorded in the same
     .ppd file. This is the cleanest option - licks and photometry share one
     clock, so no alignment is needed.
  2. A CSV of lick timestamps (one column of times).
  3. A CSV with a binary lick state column sampled over time.

Bout detection follows the usual rodent convention: mice lick in rhythmic bursts
at roughly 6-10 Hz, so consecutive licks separated by less than an inter-bout
threshold belong to the same bout, and a bout must contain a minimum number of
licks to count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from photometry_core import zscore

__all__ = [
    "LICK_DEFAULTS",
    "licks_from_digital",
    "licks_from_timestamp_csv",
    "licks_from_state_csv",
    "detect_bouts",
    "lick_rate_trace",
    "peri_event_matrix",
    "bout_summary_table",
]


LICK_DEFAULTS = {
    "min_inter_lick_sec": 0.05,    # debounce: licks closer than this are one contact
    "inter_bout_sec": 1.0,         # gap that separates two bouts
    "min_licks_per_bout": 3,       # ignore isolated stray contacts
    "rate_window_sec": 60.0,       # window for the lick-rate trace
    "pre_sec": 10.0,               # peri-event window before onset
    "post_sec": 20.0,              # peri-event window after onset
    "baseline_start_sec": -10.0,   # baseline epoch for peri-event z-scoring
    "baseline_end_sec": -2.0,
}


# =============================================================================
# INGESTION
# =============================================================================

def _debounce(times, min_gap):
    """Collapse contacts closer together than min_gap into a single lick."""
    times = np.sort(np.asarray(times, dtype=float))
    times = times[np.isfinite(times)]
    if times.size == 0 or min_gap <= 0:
        return times
    keep = [times[0]]
    for t in times[1:]:
        if t - keep[-1] >= min_gap:
            keep.append(t)
    return np.asarray(keep, dtype=float)


def licks_from_digital(digital, fs, min_inter_lick_sec=0.05, time_offset=0.0):
    """
    Lick times from a pyPhotometry digital channel.

    Takes rising edges, so one lick = one low->high transition regardless of how
    long the contact is held.
    """
    d = np.asarray(digital).astype(np.int8)
    if d.size < 2:
        return np.array([], dtype=float)
    rising = np.flatnonzero(np.diff(d) == 1) + 1
    times = rising.astype(float) / float(fs) + float(time_offset)
    return _debounce(times, min_inter_lick_sec)


def licks_from_timestamp_csv(df, time_column=None, unit="s",
                             min_inter_lick_sec=0.05):
    """Lick times from a CSV holding one timestamp per lick."""
    if time_column is None:
        for c in ["lick_time", "lick_times", "timestamp", "time_sec", "time",
                  "Time", "onset", "licks"]:
            if c in df.columns:
                time_column = c
                break
        else:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                raise ValueError("No numeric column found for lick timestamps.")
            time_column = numeric[0]

    times = pd.to_numeric(df[time_column], errors="coerce").dropna().to_numpy(float)
    if unit == "ms":
        times = times / 1000.0
    elif unit == "min":
        times = times * 60.0
    return _debounce(times, min_inter_lick_sec), time_column


def licks_from_state_csv(df, time_column, state_column, unit="s",
                         min_inter_lick_sec=0.05):
    """Lick times from a sampled binary state column (0/1 or False/True)."""
    t = pd.to_numeric(df[time_column], errors="coerce").to_numpy(float)
    s = pd.to_numeric(df[state_column], errors="coerce").fillna(0).to_numpy()
    if unit == "ms":
        t = t / 1000.0
    elif unit == "min":
        t = t * 60.0

    good = np.isfinite(t) & np.isfinite(s)
    t, s = t[good], (s[good] > 0).astype(np.int8)
    if t.size < 2:
        return np.array([], dtype=float)
    rising = np.flatnonzero(np.diff(s) == 1) + 1
    return _debounce(t[rising], min_inter_lick_sec)


# =============================================================================
# BOUT DETECTION
# =============================================================================

def detect_bouts(lick_times, inter_bout_sec=1.0, min_licks_per_bout=3):
    """
    Group licks into bouts.

    Returns a DataFrame with one row per bout: onset, offset, duration,
    n_licks, mean within-bout lick rate (Hz), and the gap since the previous bout.
    """
    lick_times = np.sort(np.asarray(lick_times, dtype=float))
    lick_times = lick_times[np.isfinite(lick_times)]
    cols = ["bout", "onset_sec", "offset_sec", "duration_sec", "n_licks",
            "lick_rate_hz", "gap_before_sec"]
    if lick_times.size == 0:
        return pd.DataFrame(columns=cols)

    gaps = np.diff(lick_times)
    breaks = np.flatnonzero(gaps > float(inter_bout_sec))
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [lick_times.size - 1]])

    rows = []
    prev_offset = None
    for s, e in zip(starts, ends):
        n = int(e - s + 1)
        if n < int(min_licks_per_bout):
            continue
        onset, offset = float(lick_times[s]), float(lick_times[e])
        dur = offset - onset
        rows.append({
            "onset_sec": onset,
            "offset_sec": offset,
            "duration_sec": dur,
            "n_licks": n,
            # (n-1) intervals span the bout, so rate = (n-1)/duration.
            "lick_rate_hz": (n - 1) / dur if dur > 0 else np.nan,
            "gap_before_sec": np.nan if prev_offset is None else onset - prev_offset,
        })
        prev_offset = offset

    out = pd.DataFrame(rows, columns=[c for c in cols if c != "bout"])
    out.insert(0, "bout", np.arange(1, len(out) + 1))
    return out


def lick_rate_trace(lick_times, time_grid, window_sec=60.0):
    """Lick rate (Hz) on a supplied time grid, via a centred counting window."""
    time_grid = np.asarray(time_grid, dtype=float)
    lick_times = np.sort(np.asarray(lick_times, dtype=float))
    if lick_times.size == 0 or time_grid.size == 0:
        return np.zeros_like(time_grid)
    half = float(window_sec) / 2.0
    left = np.searchsorted(lick_times, time_grid - half, side="left")
    right = np.searchsorted(lick_times, time_grid + half, side="right")
    return (right - left) / float(window_sec)


# =============================================================================
# PERI-EVENT ALIGNMENT
# =============================================================================

def peri_event_matrix(time_sec, values, event_times, pre_sec=10.0, post_sec=20.0,
                      baseline_start_sec=-10.0, baseline_end_sec=-2.0,
                      normalise="baseline_z", target_dt=None):
    """
    Build a trials x time matrix of the signal aligned to each event.

    normalise
      "baseline_z" : z-score each trial against its own pre-event baseline.
                     This is the standard for event-locked photometry - the units
                     become "SDs away from this animal's own pre-event state",
                     which is comparable across trials, animals and sessions.
      "baseline_sub": subtract the baseline mean only (keeps dF/F units).
      "none"        : raw values.

    Trials whose full window falls outside the recording are dropped, so a
    partial window can never be silently padded with zeros.

    Returns (matrix, trial_time_axis, kept_event_times).
    """
    time_sec = np.asarray(time_sec, dtype=float)
    values = np.asarray(values, dtype=float)
    event_times = np.asarray(event_times, dtype=float)

    if time_sec.size < 2 or event_times.size == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)

    if target_dt is None:
        target_dt = float(np.median(np.diff(time_sec)))
    if not np.isfinite(target_dt) or target_dt <= 0:
        target_dt = 0.1

    trial_t = np.arange(-abs(pre_sec), abs(post_sec) + target_dt, target_dt)
    t_min, t_max = float(time_sec[0]), float(time_sec[-1])

    order = np.argsort(time_sec)
    ts, vs = time_sec[order], values[order]

    rows, kept = [], []
    for ev in event_times:
        if ev - abs(pre_sec) < t_min or ev + abs(post_sec) > t_max:
            continue
        trace = np.interp(ev + trial_t, ts, vs)

        if normalise in ("baseline_z", "baseline_sub"):
            bmask = (trial_t >= baseline_start_sec) & (trial_t <= baseline_end_sec)
            if bmask.sum() < 2:
                bmask = trial_t < 0
            if bmask.sum() < 2:
                continue
            if normalise == "baseline_z":
                trace = zscore(trace, baseline_mask=bmask)
            else:
                trace = trace - np.nanmean(trace[bmask])

        rows.append(trace)
        kept.append(ev)

    if not rows:
        return np.zeros((0, trial_t.size)), trial_t, np.zeros(0)
    return np.vstack(rows), trial_t, np.asarray(kept, dtype=float)


def bout_summary_table(bouts, lick_times, total_duration_sec):
    """Session-level lickometer summary."""
    n_licks = int(np.asarray(lick_times).size)
    hours = total_duration_sec / 3600.0 if total_duration_sec else np.nan
    if bouts is None or bouts.empty:
        return pd.DataFrame([{
            "Total licks": n_licks,
            "Licks per hour": round(n_licks / hours, 2) if hours else np.nan,
            "Bouts": 0, "Mean licks per bout": np.nan,
            "Mean bout duration (s)": np.nan,
            "Mean within-bout rate (Hz)": np.nan,
            "Time spent licking (%)": 0.0,
        }])
    return pd.DataFrame([{
        "Total licks": n_licks,
        "Licks per hour": round(n_licks / hours, 2) if hours else np.nan,
        "Bouts": int(len(bouts)),
        "Mean licks per bout": round(float(bouts.n_licks.mean()), 2),
        "Mean bout duration (s)": round(float(bouts.duration_sec.mean()), 2),
        "Mean within-bout rate (Hz)": round(float(bouts.lick_rate_hz.mean()), 2),
        "Time spent licking (%)": round(
            100.0 * float(bouts.duration_sec.sum()) / total_duration_sec, 2
        ) if total_duration_sec else np.nan,
    }])
