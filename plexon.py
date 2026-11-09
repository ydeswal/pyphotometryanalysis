#!/usr/bin/env python3
"""
plexon.py
=========

Reading CSV exports of the Plexon Multi-Wavelength Photometry System (PMPS)
and the event lists exported alongside them.

Photometry CSV (one row per camera frame, ~30 Hz):

    Frame,timestamp,RawF_465_F1, F0_465_F1, R_465_F1,RawF_410_F1, F0_410_F1, R_410_F1,RawF_560_F1, ...

  RawF_<nm>_F<k>   raw fluorescence of fibre k excited at <nm> nm (camera counts)
  F0_<nm>_F<k>     Plexon's own running baseline of that channel
  R_<nm>_F<k>      Plexon's own (RawF - F0) / F0
  timestamp        seconds since the recording started (no clock time is stored)

The app uses RawF of the signal wavelength (465 nm, GCaMP) and of the
isosbestic control (410 nm on Plexon systems) and runs its own pipeline on them,
exactly as for pyPhotometry data. Plexon's R columns are kept for reference.
A 560 nm channel with no red indicator is just background.

Event CSV:

    Event,"Start Time (s)","End Time (s)"
    "drop",115.9,116.03
    "eating",250.5,305.2

Event files are matched to recordings by name: '8786N-events.csv',
'8781R=events.csv' and '8784Revents.csv' belong to '8786N.csv', '8781R.csv' and
'8784-R.csv'.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

_RAW = re.compile(r"^\s*RawF_(\d+)_F(\d+)\s*$", re.I)
EVENT_COLORS = {"drop": "#dc2626", "food": "#dc2626", "eating": "#f59e0b", "eat": "#f59e0b",
                "drink": "#0ea5e9", "lick": "#0ea5e9"}
_PALETTE = ["#7c3aed", "#0891b2", "#db2777", "#65a30d", "#ea580c", "#4f46e5"]


def header_of(path, n=2):
    try:
        with open(path, "r", errors="replace") as fh:
            return [fh.readline() for _ in range(n)]
    except OSError:
        return []


def _cols(header_line):
    return [c.strip().strip('"') for c in str(header_line).strip().split(",")]


def channels(columns):
    """{fibre: [wavelengths]} from the column names, e.g. {'F1': ['465', '410', '560']}."""
    out = {}
    for c in columns:
        m = _RAW.match(str(c))
        if m:
            out.setdefault(f"F{m.group(2)}", []).append(m.group(1))
    return out


def looks_like_photometry(columns):
    cols = [str(c).strip() for c in columns]
    return bool(channels(cols)) and any(c.lower() == "timestamp" for c in cols)


def looks_like_events(columns):
    cols = [str(c).strip().strip('"').lower() for c in columns]
    return "event" in cols and any(c.startswith("start time") for c in cols)


def sniff(path):
    """'photometry', 'events' or None for one CSV file."""
    h = header_of(path, 1)
    if not h:
        return None
    cols = _cols(h[0])
    if looks_like_photometry(cols):
        return "photometry"
    if looks_like_events(cols):
        return "events"
    return None


def default_pair(waves):
    """(signal, control) wavelengths: 465 (or the one nearest 470) and 410/405 (nearest 410)."""
    w = sorted(set(waves), key=float)
    sig = min(w, key=lambda x: abs(float(x) - 470))
    rest = [x for x in w if x != sig] or w
    ctl = min(rest, key=lambda x: abs(float(x) - 410))
    return sig, ctl


def to_two_channel(df, roi, fibre="F1", signal_nm=None, control_nm=None):
    """
    Plexon frame table -> time_sec, <roi>_sig_raw, <roi>_uv_raw (+ Plexon's own R columns as
    plexon_R_<nm>), ready for the app's raw-CSV pipeline. Returns (table, info dict).
    """
    df = df.rename(columns=lambda c: str(c).strip())
    ch = channels(df.columns)
    if fibre not in ch:
        raise ValueError(f"No fibre {fibre} in this file (found {', '.join(ch) or 'none'}).")
    s_def, c_def = default_pair(ch[fibre])
    sig = str(signal_nm or s_def)
    ctl = str(control_nm or c_def)
    for w in (sig, ctl):
        if w not in ch[fibre]:
            raise ValueError(f"No {w} nm channel for {fibre} (found {', '.join(ch[fibre])} nm).")
    t = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(float)
    out = pd.DataFrame({
        "time_sec": t - (t[np.isfinite(t)][0] if np.isfinite(t).any() else 0.0),
        f"{roi}_sig_raw": pd.to_numeric(df[f"RawF_{sig}_{fibre}"], errors="coerce").to_numpy(float),
        f"{roi}_uv_raw": pd.to_numeric(df[f"RawF_{ctl}_{fibre}"], errors="coerce").to_numpy(float),
    })
    keep = np.isfinite(out.to_numpy(float)).all(axis=1)
    for w in ch[fibre]:
        if f"R_{w}_{fibre}" in df.columns:
            out[f"plexon_R_{w}"] = pd.to_numeric(df[f"R_{w}_{fibre}"], errors="coerce").to_numpy(float)
    out = out.loc[keep].reset_index(drop=True)
    dt = np.diff(out["time_sec"].to_numpy())
    fs = 1.0 / np.median(dt[dt > 0]) if (dt > 0).any() else float("nan")
    gaps = int((dt > 1.5 / fs).sum()) if np.isfinite(fs) else 0
    info = {"input_type": "plexon_csv", "plexon_fibre": fibre, "plexon_signal_nm": sig, "plexon_control_nm": ctl,
            "plexon_all_nm": ch[fibre], "plexon_fibres": list(ch), "plexon_frame_gaps": gaps,
            "plexon_fs": float(fs)}
    return out, info


def notes_for(info):
    """Plain-language processing notes for a Plexon recording."""
    n = [f"Plexon PMPS file: fibre {info['plexon_fibre']}, signal {info['plexon_signal_nm']} nm, control "
         f"{info['plexon_control_nm']} nm, {info['plexon_fs']:.1f} frames/s. Raw fluorescence (RawF) was run "
         "through the same pipeline as pyPhotometry data; Plexon's own ΔF/F (R columns) is kept as plexon_R_<nm>.",
         "Plexon files store time from the start of the recording only, not the clock time, so clock-time "
         "features (FED alignment, time of day) are not available for them."]
    if str(info.get("plexon_control_nm")) == "410":
        n.append("The control is 410 nm, closer to GCaMP's isosbestic point than 405 nm, so it should barely "
                 "move with calcium. Where the app says '405', read 'the control channel'.")
    if info.get("plexon_frame_gaps"):
        n.append(f"{info['plexon_frame_gaps']} gap(s) in the frame timestamps (dropped camera frames).")
    return n


def read_events(path_or_df):
    """-> list of {'label', 'start', 'end'} (seconds since the recording started)."""
    df = pd.read_csv(path_or_df) if not isinstance(path_or_df, pd.DataFrame) else path_or_df
    df = df.rename(columns=lambda c: str(c).strip().strip('"'))
    lab = next(c for c in df.columns if c.lower() == "event")
    st_ = next(c for c in df.columns if c.lower().startswith("start time"))
    en = next((c for c in df.columns if c.lower().startswith("end time")), None)
    out = []
    for _, r in df.iterrows():
        try:
            a = float(r[st_])
        except (TypeError, ValueError):
            continue
        try:
            b = float(r[en]) if en is not None else None
        except (TypeError, ValueError):
            b = None
        out.append({"label": str(r[lab]).strip() or "event", "start": a,
                    "end": b if b is not None and np.isfinite(b) and b > a else None})
    out.sort(key=lambda e: e["start"])
    return out


def _key(name):
    s = Path(str(name)).stem.lower()
    s = re.sub(r"events?", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def match_events(event_names, recording_names):
    """{event file name: recording name or None}, by name with 'events' and punctuation removed."""
    keys = {}
    for r in recording_names:
        keys.setdefault(_key(r), r)
    return {e: keys.get(_key(e)) for e in event_names}


def colour_for(label, i=0):
    low = str(label).lower()
    for k, v in EVENT_COLORS.items():
        if k in low:
            return v
    return _PALETTE[i % len(_PALETTE)]
