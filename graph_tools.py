#!/usr/bin/env python3
"""
graph_tools.py
==============

Display-side tools for the Streamlit app. Nothing here changes processed data
or downloads; every function works on what is drawn.

1. lighten_figure()   Min/max decimation of every line trace before it is sent
                      to the browser. A 24 h recording at 1 Hz is 86,400 points
                      per trace; the old app sent every point of every trace of
                      every tab on every rerun (81 MB for one recording), which
                      is what made the hosted app slow and then crash. Keeping
                      the first, last, minimum and maximum sample of each bin
                      preserves every peak and trough you can see on screen.

2. Peaks & troughs    Ranks the highest peaks (ceilings) and lowest troughs
                      (floors) of a dF/F or z-score trace as separate events,
                      marks the current ceiling and floor, and lets you remove
                      them one at a time. A removed event is blanked from where
                      the trace leaves its median to where it returns, and the
                      marker moves to the next highest / lowest. The y-axis can
                      refit to what is left, and a manual floor and ceiling can
                      each be set independently.

3. Time ruler         Click two points on any graph; shows both times (elapsed
                      and wall clock), the time between them, and the mean,
                      min and max of the signal in between. Uses Streamlit's
                      own Plotly click selection (no extra package), inside an
                      st.fragment so a click reruns only that graph.

4. render_workspace() 2-6 graphs side by side, each with its own recording and
                      graph type, on elapsed or wall-clock time, with an
                      optional shared time window and matched y-scales.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy.signal import find_peaks as _find_peaks
except Exception:                       # pragma: no cover
    _find_peaks = None

DETAIL_OPTIONS = {
    "Fast (4,000 points per trace)": 4_000,
    "Balanced (12,000 points per trace)": 12_000,
    "Fine (40,000 points per trace)": 40_000,
    "Every point (slow for long recordings)": 0,
}
DEFAULT_DETAIL = "Balanced (12,000 points per trace)"

_CONFIG = {"scrollZoom": False, "displaylogo": False, "responsive": True,
           "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
_CEIL_COLOR = "#c2410c"
_FLOOR_COLOR = "#1d4ed8"
_RULER_COLORS = ("#0f766e", "#7c3aed")


# =============================================================================
# FORMATTING
# =============================================================================

def fmt_hms(seconds, decimals=2):
    """Seconds -> [-]HH:MM:SS.ss"""
    if seconds is None or not np.isfinite(seconds):
        return "n/a"
    sign = "-" if seconds < 0 else ""
    s = round(abs(float(seconds)), decimals)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    width = 3 + decimals if decimals else 2
    return f"{sign}{int(h):02d}:{int(m):02d}:{sec:0{width}.{decimals}f}"


def fmt12(ts, decimals=1, date=True, seconds=True):
    """
    A wall-clock time in 12-hour form, e.g. '2026-11-04 3:57:31.6 PM'.
    Fractions of a second are cut, not rounded, so a time never shows the
    next second. date=False drops the date; seconds=False gives '3:57 PM'.
    """
    if ts is None or pd.isna(ts):
        return ""
    ts = pd.Timestamp(ts)
    h = ts.hour % 12 or 12
    out = f"{h}:{ts.minute:02d}"
    if seconds:
        out += f":{ts.second:02d}"
        if decimals:
            frac = int(ts.microsecond // 10 ** (6 - decimals))
            out += "." + str(frac).zfill(decimals)
    out += " AM" if ts.hour < 12 else " PM"
    return (ts.strftime("%Y-%m-%d ") + out) if date else out


def fmt_clock(clock_start, seconds, date=True, decimals=2):
    """Wall-clock time of `seconds` into a recording, 12-hour with AM/PM."""
    if clock_start is None or seconds is None or not np.isfinite(seconds):
        return ""
    ts = pd.Timestamp(clock_start) + pd.Timedelta(seconds=float(seconds))
    return fmt12(ts, decimals=decimals, date=date)


def iso_to_12h(values, with_date="md"):
    """
    Vectorised: ISO clock strings 'YYYY-MM-DD HH:MM:SS.fff' -> 12-hour
    strings ('MM-DD h:MM:SS.fff PM' with with_date="md", 'YYYY-MM-DD ...' with
    "ymd", time only with None). Strings that do not look like ISO times are
    passed through unchanged.
    """
    v = np.asarray(values).astype(str)
    out = v.astype(object)
    if v.size == 0:
        return out
    w = v.dtype.itemsize // 4
    if w < 19:
        return out
    ch = np.ascontiguousarray(v).view("U1").reshape(-1, w)
    good = (ch[:, 4] == "-") & (ch[:, 13] == ":") & np.isin(ch[:, 11], list("012")) \
        & np.char.isdigit(ch[:, 12])
    if not good.any():
        return out
    g = ch[good]
    hh = g[:, 11].astype(int) * 10 + g[:, 12].astype(int)
    h12 = hh % 12
    h12[h12 == 0] = 12
    rest = np.ascontiguousarray(g[:, 13:]).view(f"U{w - 13}").ravel()
    rest = np.char.rstrip(rest)
    t12 = np.char.add(np.char.add(h12.astype(str), rest), np.where(hh < 12, " AM", " PM"))
    if with_date == "md":
        pre = np.ascontiguousarray(g[:, 5:11]).view("U6").ravel()
    elif with_date == "ymd":
        pre = np.ascontiguousarray(g[:, 0:11]).view("U11").ravel()
    else:
        pre = None
    out[good] = (np.char.add(pre, t12) if pre is not None else t12).astype(object)
    return out


def parse_tod(text):
    """'7:00 PM', '7pm', '19:00', '7:30 am', '19' -> hours after midnight (float), or None."""
    import re
    t = str(text or "").strip().lower().replace(".", "")
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?(?::(\d{2}))?\s*(am|pm|a|p)?", t)
    if not m:
        return None
    h, mi, se, ap = int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0), m.group(4)
    if mi > 59 or se > 59:
        return None
    if ap:
        if not 1 <= h <= 12:
            return None
        h = h % 12 + (12 if ap.startswith("p") else 0)
    elif h > 24:
        return None
    return (h % 24) + mi / 60 + se / 3600


def fmt_tod(hours):
    """19.5 -> '7:30 PM'."""
    h = float(hours) % 24
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = (hh + 1) % 24, 0
    return f"{hh % 12 or 12}:{mm:02d} {'AM' if hh < 12 else 'PM'}"


def clock_start_of(df, settings=None):
    """Wall-clock time of time_sec = 0 for a processed DataFrame, or None."""
    if df is not None and "clock_time" in df.columns and len(df):
        try:
            first = pd.Timestamp(str(df["clock_time"].iloc[0]))
            return first - pd.Timedelta(seconds=float(df["time_sec"].iloc[0]))
        except (ValueError, TypeError):
            pass
    start = (settings or {}).get("start_clock_local")
    if start:
        try:
            ts = pd.Timestamp(start)
            off = (settings or {}).get("crop_start_sec_original_timeline") \
                if (settings or {}).get("cropped") else 0.0
            return ts + pd.Timedelta(seconds=float(off or 0.0))
        except (ValueError, TypeError):
            return None
    return None


_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200,
          10800, 14400, 21600, 43200, 86400, 172800]


def time_ticks(xmin, xmax, clock_start=None, target=8):
    """Tick positions (seconds) and labels: elapsed HH:MM:SS or wall clock."""
    if not (np.isfinite(xmin) and np.isfinite(xmax)) or xmax <= xmin:
        return None, None
    span = xmax - xmin
    step = next((s for s in _STEPS if span / s <= target), _STEPS[-1])
    if clock_start is None:
        first = math.ceil(xmin / step) * step
        vals = np.arange(first, xmax + 1e-9, step)
        return vals, [fmt_hms(v, 0) for v in vals]
    # Wall clock: ticks on round clock times, e.g. every 2 h at 10:00, 12:00...
    c0 = pd.Timestamp(clock_start)
    abs0 = (c0 - c0.normalize()).total_seconds() + xmin
    first = math.ceil(abs0 / step) * step
    vals = np.arange(first, abs0 + span + 1e-9, step) - abs0 + xmin
    labels = []
    for v in vals:
        ts = (c0 + pd.Timedelta(seconds=float(v))).round("ms")   # 11:59:59.999999 -> 12:00
        lab = fmt12(ts, decimals=0, date=False, seconds=step < 60)
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0 or v == vals[0]:
            lab = ts.strftime("%b %d<br>") + lab
        labels.append(lab)
    return vals, labels


# =============================================================================
# DECIMATION
# =============================================================================

def minmax_indices(y, max_points):
    """
    Indices that keep, for each of ~max_points/4 bins, the first, last,
    minimum and maximum finite sample, plus one NaN where a bin has a gap (so
    blanked regions stay blank instead of being bridged by a line).
    Returns None when no decimation is needed.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if not max_points or n <= max_points:
        return None
    nb = max(1, int(max_points) // 4)
    b = int(math.ceil(n / nb))
    nb = int(math.ceil(n / b))
    pad = nb * b - n
    yp = np.concatenate([y, np.full(pad, np.nan)]) if pad else y
    blk = yp.reshape(nb, b)
    finite = np.isfinite(blk)
    any_f = finite.any(axis=1)
    base = np.arange(nb) * b
    hi = np.where(finite, blk, -np.inf).argmax(axis=1)
    lo = np.where(finite, blk, np.inf).argmin(axis=1)
    first_f = finite.argmax(axis=1)
    last_f = b - 1 - finite[:, ::-1].argmax(axis=1)
    has_nan = (~finite).any(axis=1)
    first_nan = (~finite).argmax(axis=1)
    parts = [base[any_f] + hi[any_f], base[any_f] + lo[any_f],
             base[any_f] + first_f[any_f], base[any_f] + last_f[any_f],
             base[has_nan] + first_nan[has_nan], base[~any_f]]
    idx = np.unique(np.concatenate(parts))
    return idx[idx < n]


def lighten_figure(fig, max_points):
    """Decimate every long line trace in place. Marker-only traces (lick
    rasters) and filled shapes (bout bands) are left alone."""
    if not max_points:
        return fig
    for tr in fig.data:
        if tr.type not in ("scatter", "scattergl"):
            continue
        mode = tr.mode or "lines"
        # Fills that close a shape (bout bands, SEM ribbons) depend on point
        # order and are left intact; fills to zero decimate safely.
        if "lines" not in mode or tr.fill in ("toself", "tonext", "tonexty", "tonextx"):
            continue
        if tr.x is None or tr.y is None:
            continue
        n = len(tr.y)
        if n <= max_points or len(tr.x) != n:
            continue
        try:
            yy = np.asarray(tr.y, dtype=float)
        except (TypeError, ValueError):
            continue
        idx = minmax_indices(yy, max_points)
        if idx is None:
            continue
        upd = {"x": np.asarray(tr.x)[idx], "y": yy[idx]}
        for attr in ("customdata", "text", "hovertext"):
            v = getattr(tr, attr)
            if v is not None and not isinstance(v, str) and len(v) == n:
                upd[attr] = np.asarray(v)[idx]
        tr.update(upd)
    return fig


# =============================================================================
# PEAKS AND TROUGHS
# =============================================================================

def rank_extremes(t, y, n=10, min_sep_sec=60.0, highest=True):
    """
    The n highest peaks (or lowest troughs) as DISTINCT events: local extrema
    ranked by value, keeping one only if it is at least min_sep_sec from every
    event already kept (so one broad transient is not listed ten times).
    Returns a DataFrame: rank, idx, time_sec, value.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    cols = ["rank", "idx", "time_sec", "value"]
    if ok.sum() == 0:
        return pd.DataFrame(columns=cols)
    v = np.where(ok, y if highest else -y, -np.inf)
    if _find_peaks is not None:
        cand = _find_peaks(v)[0]
    else:
        mid = (v[1:-1] >= v[:-2]) & (v[1:-1] > v[2:])
        cand = np.flatnonzero(mid) + 1
    fin = np.flatnonzero(ok)
    cand = np.unique(np.concatenate([cand, [fin[0], fin[-1], int(np.argmax(v))]]))
    cand = cand[np.isfinite(v[cand])]
    order = cand[np.argsort(-v[cand], kind="stable")]
    picked = []
    for i in order:
        if all(abs(t[i] - t[p]) >= min_sep_sec for p in picked):
            picked.append(int(i))
            if len(picked) >= n:
                break
    out = pd.DataFrame({"idx": picked, "time_sec": t[picked], "value": y[picked]})
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def excursion_bounds(t, y, i, baseline, max_half_sec):
    """
    Time span of the excursion containing sample i: outward from i while the
    trace stays on the same side of `baseline`, at most max_half_sec each way.
    This is what "removing a peak" blanks, so a broad transient goes as a
    whole and a one-sample spike takes only itself.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    sign = 1.0 if y[i] >= baseline else -1.0
    lo_lim = t[i] - max_half_sec
    hi_lim = t[i] + max_half_sec
    j0 = int(np.searchsorted(t, lo_lim, side="left"))
    j1 = int(np.searchsorted(t, hi_lim, side="right"))
    seg = sign * (y[j0:j1] - baseline) > 0
    k = i - j0
    left = seg[:k + 1][::-1]
    stop = np.flatnonzero(~left)
    lo = i - (stop[0] - 1 if stop.size else k)
    right = seg[k:]
    stop = np.flatnonzero(~right)
    hi = i + (stop[0] - 1 if stop.size else right.size - 1)
    lo = max(lo, 0)
    hi = min(hi, len(t) - 1)
    return float(t[lo]), float(t[hi])


def _blank(x, y, spans):
    y = np.array(y, dtype=float, copy=True)
    x = np.asarray(x, dtype=float)
    for lo, hi in spans:
        y[(x >= lo) & (x <= hi)] = np.nan
    return y


def _trace_axes(tr):
    return (tr.xaxis or "x"), (tr.yaxis or "y")


def _line_traces(fig, min_points=20):
    out = []
    for k, tr in enumerate(fig.data):
        if tr.type not in ("scatter", "scattergl") or tr.y is None or tr.x is None:
            continue
        if "lines" not in (tr.mode or "lines") or tr.fill not in (None, "none"):
            continue
        if tr.visible in (False, "legendonly") or len(tr.y) < min_points:
            continue
        try:
            np.asarray(tr.x, dtype=float)
        except (TypeError, ValueError):
            continue
        out.append(k)
    return out


def _peak_controls(fig, key, trace_k, x_unit_sec, clock_start, default_markers, compact=False):
    """UI + figure edits for one trace. Returns (t, y_after_removal, label)."""
    tr = fig.data[trace_k]
    label = tr.name or "signal"
    t = np.asarray(tr.x, dtype=float) * x_unit_sec
    y = np.asarray(tr.y, dtype=float)
    ver_key = f"{key}_pk_ver"
    st.session_state.setdefault(ver_key, 0)

    c = st.columns(2) * 2 if compact else st.columns(4)
    show_ceil = c[0].checkbox("Mark ceiling (highest)", value=default_markers, key=f"{key}_pk_sc")
    show_floor = c[1].checkbox("Mark floor (lowest)", value=default_markers, key=f"{key}_pk_sf")
    n_list = c[2].selectbox("List top", [5, 10, 20], index=0, key=f"{key}_pk_n")
    sep = c[3].number_input("Min. gap between events (s)", 1.0, 36000.0, 60.0, 10.0,
                            key=f"{key}_pk_sep",
                            help="Two extremes closer than this count as one event.")
    c2 = st.columns(2) + [st] if compact else st.columns([1, 1, 2])
    max_half = c2[0].number_input("Removal reaches at most ± (s)", 0.5, 36000.0, 60.0, 5.0,
                                  key=f"{key}_pk_half",
                                  help="A removed event is blanked from where the trace "
                                       "leaves its median to where it returns, but never "
                                       "further than this either side.")
    refit = c2[1].checkbox("Refit y-axis to what's left", value=True, key=f"{key}_pk_refit")
    if c2[2].button("Restore all removed", key=f"{key}_pk_reset"):
        st.session_state[ver_key] += 1
    ver = st.session_state[ver_key]

    hi = rank_extremes(t, y, n_list, sep, highest=True)
    lo = rank_extremes(t, y, n_list, sep, highest=False)

    def _table(df):
        return pd.DataFrame({
            "Remove": False, "#": df["rank"].to_numpy(),
            "Value": df["value"].round(4).to_numpy(),
            "Elapsed": [fmt_hms(v) for v in df["time_sec"]],
            "Clock": [fmt_clock(clock_start, v) for v in df["time_sec"]],
        })

    cfg = {"Remove": st.column_config.CheckboxColumn(
        "Remove", help="Blank this event from the graph; the marker moves to the next one.")}
    e1, e2 = (st.container(), st.container()) if compact else st.columns(2)
    with e1:
        st.markdown(f"**Highest peaks (ceilings)** · {label}")
        ed_hi = st.data_editor(_table(hi), key=f"{key}_pk_hi_{ver}", hide_index=True,
                               disabled=["#", "Value", "Elapsed", "Clock"], column_config=cfg,
                               use_container_width=True)
    with e2:
        st.markdown(f"**Lowest troughs (floors)** · {label}")
        ed_lo = st.data_editor(_table(lo), key=f"{key}_pk_lo_{ver}", hide_index=True,
                               disabled=["#", "Value", "Elapsed", "Clock"], column_config=cfg,
                               use_container_width=True)

    finite = y[np.isfinite(y)]
    baseline = float(np.median(finite)) if finite.size else 0.0
    spans = []
    for table, ed in ((hi, ed_hi), (lo, ed_lo)):
        for r, rem in enumerate(ed["Remove"].to_numpy()):
            if rem and r < len(table):
                spans.append(excursion_bounds(t, y, int(table["idx"].iloc[r]), baseline, max_half))
    y2 = _blank(t, y, spans) if spans else y
    if spans:
        tr.update(y=y2)

    # Current ceiling/floor = extremes of what is left.
    ok = np.isfinite(y2) & np.isfinite(t)
    xa, ya = _trace_axes(tr)
    if ok.any():
        i_max = int(np.flatnonzero(ok)[np.argmax(y2[ok])])
        i_min = int(np.flatnonzero(ok)[np.argmin(y2[ok])])
        for show, i, name, col, sym in ((show_ceil, i_max, "Ceiling", _CEIL_COLOR, "triangle-down"),
                                        (show_floor, i_min, "Floor", _FLOOR_COLOR, "triangle-up")):
            if not show:
                continue
            fig.add_trace(go.Scatter(
                x=[t[i] / x_unit_sec], y=[y2[i]], mode="markers+text", xaxis=xa, yaxis=ya,
                marker=dict(size=13, color=col, symbol=sym, line=dict(color="#fff", width=1)),
                text=[f"{name} {y2[i]:.3g}"],
                textposition="middle right", cliponaxis=False,
                textfont=dict(color=col, size=11), name=f"{name} ({label})", showlegend=False,
                hovertemplate=(f"{name}: %{{y:.5g}}<br>{fmt_hms(t[i])}"
                               + (f"<br>{fmt_clock(clock_start, t[i])}" if clock_start is not None else "")
                               + "<extra></extra>"),
            ))
        msg = (f"Current ceiling **{y2[i_max]:.4g}** at {fmt_hms(t[i_max])}"
               + (f" ({fmt_clock(clock_start, t[i_max], date=False)})" if clock_start is not None else "")
               + f" · floor **{y2[i_min]:.4g}** at {fmt_hms(t[i_min])}"
               + (f" ({fmt_clock(clock_start, t[i_min], date=False)})" if clock_start is not None else ""))
        if spans:
            msg += f" · {len(spans)} event(s) removed from the display"
        st.caption(msg)

    # Y-axis: refit, and independent manual floor / ceiling.
    m = st.columns(2) * 2 if compact else st.columns(4)
    fl_on = m[0].checkbox("Fix y floor", key=f"{key}_pk_flon")
    # Keys include the removal count so the suggested values follow what is left.
    fl = m[1].number_input("Floor", value=float(np.nanmin(y2)) if ok.any() else 0.0,
                           format="%.4f", disabled=not fl_on, key=f"{key}_pk_fl_{len(spans)}")
    ce_on = m[2].checkbox("Fix y ceiling", key=f"{key}_pk_ceon")
    ce = m[3].number_input("Ceiling", value=float(np.nanmax(y2)) if ok.any() else 1.0,
                           format="%.4f", disabled=not ce_on, key=f"{key}_pk_ce_{len(spans)}")
    if ok.any() and (fl_on or ce_on or (refit and spans)):
        lo_v, hi_v = float(np.nanmin(y2)), float(np.nanmax(y2))
        pad = 0.06 * (hi_v - lo_v if hi_v > lo_v else max(abs(hi_v), 1.0))
        rng = [fl if fl_on else lo_v - pad, ce if ce_on else hi_v + pad]
        if rng[0] < rng[1]:
            fig.layout["yaxis" + ya[1:]].update(range=rng, autorange=False)
        else:
            st.error("The y ceiling must be above the y floor.")
    st.caption("Display only: removed events and fixed axes change what is drawn, not the "
               "processed data, z-scores or downloads.")
    return t, y2, label


# =============================================================================
# CHART BLOCK: peaks + time ruler, rerun on its own
# =============================================================================

def hover_time_strings(x, clock_start=None, show_recording=True):
    """
    Two string arrays for x in seconds: wall clock 'h:MM:SS.s AM' and time into
    the recording 'HH:MM:SS.s' (hours keep counting past 24). Vectorised, so
    it is quick even for every point of a long recording.
    """
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    ms = np.where(ok, np.round(x * 1000.0), 0).astype("int64")
    n = x.size

    def _hms_part(dt64, start, stop):
        s = np.datetime_as_string(dt64, unit="ms")          # YYYY-MM-DDTHH:MM:SS.mmm
        if s.dtype.itemsize // 4 != 23:                       # unusual year; slow path
            return np.array([v[start:stop] for v in s.tolist()], dtype=object)
        return np.ascontiguousarray(s.view("U1").reshape(-1, 23)[:, start:stop]) \
            .view(f"U{stop - start}").ravel()

    clock = np.full(n, "", dtype=object)
    if clock_start is not None:
        c0 = pd.Timestamp(clock_start)
        if c0.tzinfo is not None:
            c0 = c0.tz_localize(None)
        c0 = np.datetime64(c0.to_datetime64(), "ms")
        hms = _hms_part(c0 + ms.astype("timedelta64[ms]"), 11, 21)      # HH:MM:SS.s
        ch = np.ascontiguousarray(np.asarray(hms).astype("U10")).view("U1").reshape(-1, 10)
        hh = ch[:, 0].astype(int) * 10 + ch[:, 1].astype(int)
        h12 = hh % 12
        h12[h12 == 0] = 12
        rest = np.ascontiguousarray(ch[:, 2:]).view("U8").ravel()
        clock = np.char.add(np.char.add(h12.astype(str), rest),
                            np.where(hh < 12, " AM", " PM")).astype(object)
    rec = np.full(n, "", dtype=object)
    if show_recording:
        a = np.abs(ms)
        hours = (a // 3_600_000).astype(str)
        hours = np.char.zfill(hours, 2)
        mmss = _hms_part(np.datetime64(0, "ms") + (a % 3_600_000).astype("timedelta64[ms]"), 14, 21)
        rec = np.char.add(np.char.add(np.where(ms < 0, "-", ""), np.char.add(hours, ":")),
                          mmss.astype(str)).astype(object)
    clock[~ok] = ""
    rec[~ok] = ""
    return clock, rec


def add_time_hover(fig, clock_start=None, show_recording=True):
    """
    Put the wall-clock time and the time into the recording at the top of
    the hover box of every subplot: the first line or bar trace on each
    x-axis gets them as its first hover line, and the unified hover title
    (which would only show seconds) is blanked. Call after decimation.
    """
    import re
    if clock_start is None and not show_recording:
        return fig
    done = set()
    for tr in fig.data:
        if tr.type not in ("scatter", "scattergl", "bar") or tr.x is None or tr.y is None:
            continue
        if tr.visible in (False, "legendonly"):
            continue
        if tr.type != "bar" and "lines" not in (tr.mode or "lines"):
            continue
        if tr.type != "bar" and tr.fill not in (None, "none"):
            continue
        if tr.type == "bar" and tr.orientation == "h":
            continue                       # timeline bars: x holds durations, not times
        ax = tr.xaxis or "x"
        if ax in done:
            continue
        try:
            x = np.asarray(tr.x, dtype=float)
        except (TypeError, ValueError):
            continue
        if x.size == 0 or len(tr.y) != x.size:
            continue
        clock, rec = hover_time_strings(x, clock_start, show_recording)
        old = tr.customdata
        cols = [clock, rec]
        shift = 2
        if old is not None and not isinstance(old, str) and len(old) == x.size:
            o = np.asarray(old, dtype=object)
            o = o.reshape(x.size, -1)
            cols += [o[:, j] for j in range(o.shape[1])]
        cd = np.empty((x.size, len(cols)), dtype=object)
        for j, c in enumerate(cols):
            cd[:, j] = c
        head = []
        if clock_start is not None:
            head.append("<b>%{customdata[0]}</b>")
        if show_recording:
            head.append("recording %{customdata[1]}")
        body = tr.hovertemplate
        if body:
            body = re.sub(r"customdata\[(\d+)\]", lambda m: f"customdata[{int(m.group(1)) + shift}]", body)
            body = body.replace("<extra></extra>", "")
        else:
            body = f"{tr.name or 'value'}: %{{y:.4g}}"
        tr.update(customdata=cd, hovertemplate=" · ".join(head) + "<br>" + body + "<extra></extra>")
        done.add(ax)
    if done:
        fig.update_xaxes(unifiedhovertitle=dict(text=" "))
    return fig


@st.fragment
def chart_block(fig, key, *, max_points=12_000, clock_start=None, x_unit_sec=1.0,
                peaks=False, peak_default_on=False, config=None, ruler=True, compact=False,
                event_counts=None, time_hover=None):
    """
    Show one Plotly figure with an optional peaks panel and two-click time
    ruler. event_counts: optional list of (label, times_sec, grams_each or
    None); the ruler then also reports how many of those events fall
    between its two points (e.g. pellets eaten). time_hover: None, or
    "clock+recording" / "clock" to put those times at the top of the hover
    box (x must be seconds). Runs as a fragment: clicking the graph or changing these controls
    reruns only this block, not the whole app.
    """
    fig = go.Figure(fig)                 # never mutate the caller's figure
    st.session_state.setdefault(f"{key}_pts", [])

    bar = st.columns([1, 1] if compact else [1, 1, 3])
    ruler_on = bar[0].toggle("📏 Ruler", key=f"{key}_ruler",
                             help="Click two points on the graph to measure the time between them.") \
        if ruler else False
    peaks_on = bar[1].toggle("▲▼ Peaks", value=bool(peak_default_on), key=f"{key}_peaks",
                             help="Highest peaks and lowest troughs; remove them one by one.") \
        if peaks else False

    lines = _line_traces(fig)
    target_t = target_y = None
    target_label = ""
    if peaks_on and lines:
        names = [fig.data[k].name or f"trace {k}" for k in lines]
        pref = next((i for i, nm in enumerate(names)
                     if any(s in nm.lower() for s in ("z", "dff", "df/f"))), 0)
        with st.container(border=True):
            pick = st.selectbox("Trace", range(len(lines)), index=pref,
                                format_func=lambda i: names[i], key=f"{key}_pk_tr") \
                if len(lines) > 1 else 0
            target_t, target_y, target_label = _peak_controls(
                fig, key, lines[pick], x_unit_sec, clock_start, default_markers=True,
                compact=compact)
    elif ruler_on and lines:
        tr = fig.data[lines[0]]
        target_t = np.asarray(tr.x, float) * x_unit_sec
        target_y = np.asarray(tr.y, float)
        target_label = tr.name or "signal"

    pts = st.session_state[f"{key}_pts"]
    if ruler_on:
        top = st.columns([4, 1])
        top[0].caption("Click a point on the trace, then a second one. A third click "
                       "starts a new measurement. For finer timing, narrow the visual "
                       "crop window so more points are drawn.")
        if top[1].button("Clear", key=f"{key}_clear"):
            st.session_state[f"{key}_pts"] = pts = []
            st.session_state.pop(f"{key}_last", None)
        for i, p in enumerate(pts[:2]):
            fig.add_vline(x=p / x_unit_sec, line=dict(color=_RULER_COLORS[i], width=2, dash="dash"))
        if len(pts) == 2:
            fig.add_vrect(x0=min(pts) / x_unit_sec, x1=max(pts) / x_unit_sec,
                          fillcolor="rgba(15,118,110,0.08)", line_width=0, layer="below")

    lighten_figure(fig, max_points)
    if time_hover and x_unit_sec == 1.0:
        add_time_hover(fig, clock_start, show_recording="recording" in str(time_hover))
    cfg = dict(config or _CONFIG)
    if ruler_on:
        # Plotly only reports clicks on MARKERS, never on a bare line. Give
        # every line invisible markers so a click on the graph registers;
        # with "x unified" hover a click anywhere picks the time under the
        # cursor, so you do not have to hit the line itself.
        for k in _line_traces(fig, min_points=2):
            fig.data[k].update(mode="lines+markers", marker=dict(size=6, opacity=0),
                               selected=dict(marker=dict(opacity=0)),
                               unselected=dict(marker=dict(opacity=0)))
        fig.update_layout(clickmode="event+select", dragmode="pan")
        ev = st.plotly_chart(fig, use_container_width=True, config=cfg, key=f"{key}_chart",
                             on_select="rerun", selection_mode="points")
        sel = getattr(getattr(ev, "selection", None), "points", None) or \
            (ev.get("selection", {}).get("points", []) if isinstance(ev, dict) else [])
        if sel:
            p0 = sel[0]
            fp = (p0.get("curve_number"), p0.get("point_index"), p0.get("x"))
            if fp != st.session_state.get(f"{key}_last"):
                st.session_state[f"{key}_last"] = fp
                try:
                    xv = float(p0["x"]) * x_unit_sec
                except (KeyError, TypeError, ValueError):
                    xv = None
                if xv is not None and np.isfinite(xv):
                    st.session_state[f"{key}_pts"] = [xv] if len(pts) >= 2 else pts + [xv]
                    st.rerun(scope="fragment")
        _ruler_readout(pts, clock_start, target_t, target_y, target_label, event_counts)
    else:
        st.plotly_chart(fig, use_container_width=True, config=cfg, key=f"{key}_chart")


def _ruler_readout(pts, clock_start, t, y, label, event_counts=None):
    if not pts:
        return
    if len(pts) == 1:
        st.info(f"First point: **{fmt_hms(pts[0])}**"
                + (f" ({fmt_clock(clock_start, pts[0])})" if clock_start is not None else "")
                + ". Click a second point.")
        return
    a, b = sorted(pts[:2])
    d = b - a

    def _pt(v):
        return f"**{fmt_hms(v)}**" + (f" (clock {fmt_clock(clock_start, v, date=False)})"
                                      if clock_start is not None else "")
    st.markdown(f"📏 Point 1 {_pt(a)} → Point 2 {_pt(b)}  \n"
                f"**Time between: {fmt_hms(d)}**")
    extra = f"Δt = {d:,.2f} s = {d / 60:,.3f} min = {d / 3600:,.4f} h"
    if t is not None and y is not None:
        m = (t >= a) & (t <= b) & np.isfinite(y)
        if m.any():
            extra += (f" · {label} between the points: mean {np.mean(y[m]):.4g}, "
                      f"min {np.min(y[m]):.4g}, max {np.max(y[m]):.4g} ({int(m.sum())} samples)")
    for lab, times, grams in (event_counts or []):
        tt = np.asarray(times, float)
        k = int(((tt >= a) & (tt <= b)).sum())
        st.markdown(f"**{lab.capitalize()} between the points: {k}**"
                    + (f" ({k * grams:.2f} g)" if grams else "")
                    + (f" · {k / (d / 3600):.1f} per hour" if d > 0 else ""))
    st.caption(extra)


# =============================================================================
# SIDE-BY-SIDE WORKSPACE
# =============================================================================

def panel_types(df, roi):
    """Graph types available for a processed DataFrame -> (column, unit, is_peak_type)."""
    c = df.columns
    out = {}

    def first(*names):
        return next((n for n in names if n in c), None)

    dff = first(f"{roi}_dFF_percent")
    if dff:
        out["dF/F (%)"] = (dff, "%", True)
    elif first(f"{roi}_dFF"):
        out["dF/F (ratio)"] = (f"{roi}_dFF", "", True)
    if first(f"{roi}_z_dFF"):
        out["z-score (dF/F)"] = (f"{roi}_z_dFF", "z", True)
        out["z-score, running mean"] = (f"{roi}_z_dFF", "z", True)
    for label, cand, unit in (("ΔF (465 − fitted 405)", (f"{roi}_deltaF",), "V"),
                              ("465 raw", (f"{roi}_sig_raw",), "V"),
                              ("405 raw", (f"{roi}_uv_raw",), "V"),
                              ("465 filtered", (f"{roi}_sig_filt", f"{roi}_sig_filt_10Hz_lowpass"), "V"),
                              ("405 fitted to 465", (f"{roi}_uv_fit",), "V"),
                              ("F0 baseline", (f"{roi}_F0_0p001Hz",), "V")):
        col = first(*cand)
        if col:
            out[label] = (col, unit, False)
    return out


def _panel_series(df, roi, kind, smooth_sec):
    col, unit, is_peak = panel_types(df, roi)[kind]
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    if kind == "dF/F (%)" and col.endswith("_dFF"):
        y = y * 100.0
    if kind == "z-score, running mean":
        s = pd.Series(y, index=pd.to_timedelta(t - np.nanmin(t), unit="s"))
        y = s.rolling(pd.Timedelta(seconds=max(1.0, smooth_sec)), center=True,
                      min_periods=1).mean().to_numpy(float)
    return t, y, unit, is_peak


def render_workspace(results, roi, prepare_df, colors=None, max_points=12_000,
                     height=430, config=None):
    """2-6 graphs side by side; every panel has its own recording and graph."""
    if not results:
        st.info("Run an analysis first.")
        return
    names = [str(r["name"]) for r in results]
    starts = {i: clock_start_of(r["df"], r.get("settings")) for i, r in enumerate(results)}
    has_clock = all(v is not None for v in starts.values())

    c = st.columns([1, 1, 1.4, 1.4])
    n = int(c[0].number_input("Graphs", 2, 6, 2, 1, key="ws_n"))
    per_row = int(c[1].selectbox("Per row", [1, 2, 3], index=1, key="ws_cols"))
    axis = c[2].radio("Time axis", ["Elapsed", "Clock time"] if has_clock else ["Elapsed"],
                      horizontal=True, key="ws_axis",
                      help="Clock time lines recordings up by time of day; elapsed by "
                           "time since each recording started.")
    import signal_check as _sc
    _dur = max(float(pd.to_numeric(r["df"]["time_sec"], errors="coerce").max()
                     - pd.to_numeric(r["df"]["time_sec"], errors="coerce").min()) for r in results)
    _on, _w, _why = _sc.recommend_smoothing(_dur)
    smooth_min = c[3].number_input("Running-mean window (min)", 0.05, 240.0, float(round(_w / 60.0, 2)), 0.5,
                                   key=f"ws_smooth_{_w:g}",
                                   help="Used by the 'running mean' graph types. Recommended for these "
                                        "recordings: " + _why + " " + _sc.SMOOTHING_HELP)

    o = st.columns([1.2, 1.2, 2])
    link = o[0].checkbox("Same time window in every graph", value=True, key="ws_link")
    match_y = o[1].checkbox("Same y-scale for the same graph type", value=True, key="ws_matchy")

    # Shared time window (elapsed seconds, or absolute clock time).
    win = None
    if link:
        if axis == "Clock time":
            lo = min(starts[i] + pd.Timedelta(seconds=float(results[i]["df"]["time_sec"].min()))
                     for i in starts)
            hi = max(starts[i] + pd.Timedelta(seconds=float(results[i]["df"]["time_sec"].max()))
                     for i in starts)
            lo, hi = lo.to_pydatetime().replace(microsecond=0), hi.to_pydatetime().replace(microsecond=0)
            if hi > lo:
                win = o[2].slider("Time window (clock)", min_value=lo, max_value=hi, value=(lo, hi),
                                  format="MM-DD HH:mm", key="ws_win_clock")
        else:
            dur_h = max(float(r["df"]["time_sec"].max()) for r in results) / 3600.0
            if dur_h > 0:
                w = o[2].slider("Time window (elapsed hours)", 0.0, float(np.ceil(dur_h * 100) / 100),
                                (0.0, float(np.ceil(dur_h * 100) / 100)), 0.01, key="ws_win_el")
                win = (w[0] * 3600.0, w[1] * 3600.0)

    # Panel choices first, so matched y-scales can see every panel.
    choices = []
    for p in range(n):
        choices.append(None)
    grid = []
    for row0 in range(0, n, per_row):
        grid.append(st.columns(min(per_row, n - row0), gap="medium"))
    prepared = {}
    for p in range(n):
        col = grid[p // per_row][p % per_row]
        with col:
            st.markdown(f"**Graph {p + 1}**")
            ci = st.columns(2)
            rec = ci[0].selectbox("Recording", range(len(names)), index=p % len(names) if len(names) > 1 else 0,
                                  format_func=lambda i: names[i], key=f"ws_rec_{p}")
            if rec not in prepared:
                prepared[rec] = prepare_df(results[rec]["df"])
            df = prepared[rec]
            kinds = list(panel_types(df, roi))
            if not kinds:
                st.warning(f"No photometry columns for ROI '{roi}'.")
                continue
            default = ("dF/F (%)" if p % 2 == 0 else "z-score (dF/F)") if len(names) == 1 else \
                ("dF/F (%)" if "dF/F (%)" in kinds else kinds[0])
            kind = ci[1].selectbox("Graph", kinds, index=kinds.index(default) if default in kinds else 0,
                                   key=f"ws_kind_{p}")
            t, y, unit, is_peak = _panel_series(df, roi, kind, smooth_min * 60.0)
            cs = starts.get(rec)
            if win is not None:
                if axis == "Clock time":
                    a = (pd.Timestamp(win[0]) - cs).total_seconds()
                    b = (pd.Timestamp(win[1]) - cs).total_seconds()
                else:
                    a, b = win
                keep = (t >= a) & (t <= b)
                t, y = t[keep], y[keep]
            choices[p] = dict(rec=rec, kind=kind, t=t, y=y, unit=unit, is_peak=is_peak, cs=cs)

    # Matched y-ranges per graph type: the full min/max, so no peak is ever
    # hidden off-scale (remove outliers with the Peaks tool to zoom in).
    yr = {}
    if match_y:
        for ch in choices:
            if ch is None or not np.isfinite(ch["y"]).any():
                continue
            lo, hi = float(np.nanmin(ch["y"])), float(np.nanmax(ch["y"]))
            a, b = yr.get(ch["kind"], (lo, hi))
            yr[ch["kind"]] = (min(a, lo), max(b, hi))

    for p, ch in enumerate(choices):
        if ch is None:
            continue
        with grid[p // per_row][p % per_row]:
            if ch["t"].size == 0:
                st.warning("No data in this time window.")
                continue
            color = (colors or {}).get("dff" if "dF/F" in ch["kind"] else
                                       "z_smooth" if "z" in ch["kind"] else "sig_raw", "#0072B2")
            fig = go.Figure(go.Scatter(x=ch["t"], y=ch["y"], mode="lines", name=ch["kind"],
                                       line=dict(color=color, width=1.2)))
            cs = ch["cs"] if axis == "Clock time" else None
            vals, labels = time_ticks(float(np.nanmin(ch["t"])), float(np.nanmax(ch["t"])), cs)
            fig.update_layout(height=height, margin=dict(l=60, r=15, t=36, b=50),
                              template="plotly_white", showlegend=False, hovermode="x unified",
                              dragmode="pan", title=dict(text=f"{names[ch['rec']]} · {ch['kind']}",
                                                         font=dict(size=13)))
            fig.update_xaxes(title_text="Clock time" if cs is not None else "Elapsed (h:m:s)",
                             tickvals=vals, ticktext=labels, showgrid=True, gridcolor="#e5e7eb")
            fig.update_yaxes(title_text=f"{ch['kind']}" + (f" ({ch['unit']})" if ch["unit"] and ch["unit"] not in ch["kind"] else ""),
                             showgrid=True, gridcolor="#e5e7eb")
            if ch["kind"] in yr:
                lo, hi = yr[ch["kind"]]
                pad = 0.05 * (hi - lo if hi > lo else 1.0)
                fig.update_yaxes(range=[lo - pad, hi + pad])
            chart_block(fig, key=f"ws_{p}_{ch['rec']}_{ch['kind']}", max_points=max_points,
                        clock_start=ch["cs"], peaks=True, peak_default_on=False,
                        config=config, compact=per_row > 1)
