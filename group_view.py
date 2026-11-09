#!/usr/bin/env python3
"""
group_view.py
=============

"Group analysis" display: several recordings (e.g. an ad-lib test on five
animals, or the same animals on several days) analysed together.

  Combined          group mean and band of the signal, and of FED intake,
                    on one time axis (optionally each animal faintly)
  Each recording    every recording in its own panel on the same axis, or
                    all overlaid in one graph
  Heatmap           animals x time, signal and intake (as in long-term
                    photometry papers)
  Event response    every recording around an event you add (food in,
  & signal check    injection, ...): % change, 405 check and a grade per
                    recording, ranked, with the criteria explained
                    (signal_check.py)
  Meal-aligned      signal around meal onsets, averaged per animal first,
                    then across animals; animal-level statistics and the
                    fixed 405 / motion-correction data check
  Feeding pattern   meals, meal size, inter-meal interval, satiety ratio ...
                    per animal and group mean +/- SEM

All numbers come from group_analysis.py (also used by the AI chat).
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fed3
import fed_view
import graph_tools as gt
import group_analysis as ga
import interpret
import theme as th
import signal_check as sc

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255"]
_SIG = "#0072B2"
_FED = "#b45309"
_DARK = "rgba(15,23,42,0.055)"

VIEWS = ["Combined (group mean)", "Each recording", "Heatmap", "Event response & signal check",
         "Meal-aligned (group)", "Feeding pattern table"]
RATIO = "465 ÷ 405 ratio (raw)"
RAW_KINDS = (fed_view.RAW_BOTH, RATIO, "465 raw", "405 raw", "465 filtered")
_UV = "#009E73"
_EV = "#dc2626"
EVENT_TIME_KINDS = ["Recording time", "Clock time"]
BIN_OPTS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600]
MEAL_PRESETS = {
    "Douglass et al. 2024: gap ≥ 300 s, any size": (300.0, 1),
    "FED3 paper (2021): gap < 60 s, ≥ 5 pellets": (60.0, 5),
    "Short meals: gap < 60 s, ≥ 2 pellets": (60.0, 2),
    "Custom": None,
}

REFERENCES = (
    "Methods follow: Douglass et al. 2024, <i>Cell Metabolism</i> (pyPhotometry + Doric mini-cubes, recordings up "
    "to a month: 30-min running mean, group mean ± SEM on ZT, meals ≥ 300 s apart); Loewinger et al. 2024, "
    "<i>eLife</i> (animal as the unit for group statistics); Matikainen-Ankney et al. 2021, <i>eLife</i> (FED3: "
    "meals, pellets per hour as mean ± SEM).")


def _rgba(hex_color, a):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


# =============================================================================
# AXES
# =============================================================================

def _nice_step(span_s, target=8):
    for s in (1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400):
        if span_s / s <= target:
            return s
    return 86400 * max(1, int(span_s / 86400 / target))


def _elapsed(v, step=60):
    """Signed elapsed time for ticks: m:ss under an hour of steps < 1 min, else HH:MM."""
    sgn = "−" if v < 0 else ""
    v = abs(v)
    if step < 60:
        return f"{sgn}{int(v // 60)}:{int(round(v % 60)):02d}" if v < 3600 else \
            f"{sgn}{int(v // 3600)}:{int(v % 3600 // 60):02d}:{int(round(v % 60)):02d}"
    return f"{sgn}{int(v // 3600):02d}:{int(v % 3600 // 60):02d}"


def _elapsed_long(v):
    sgn = "−" if v < 0 else ""
    v = abs(v)
    return f"{sgn}{int(v // 3600)}:{int(v % 3600 // 60):02d}:{v % 60:04.1f}"


def axis_labeler(alignment, ref_clock, lights_on=None):
    """x seconds -> (short tick label, long hover label)."""
    if alignment.startswith("Clock"):
        def f(x):
            ts = pd.Timestamp(ref_clock) + pd.Timedelta(seconds=float(x))
            return gt.fmt12(ts, 0, date=False, seconds=False), f"{ts:%b %d} {gt.fmt12(ts, 0, date=False)}"
        return f
    if alignment.startswith("Time of day"):
        def f(x):
            day = int(x // 86400) + 1
            ts = pd.Timestamp("2000-01-01") + pd.Timedelta(seconds=float(x % 86400))
            zt = ""
            if lights_on is not None:
                z = ((x / 3600.0) - lights_on) % 24
                zt = f" · ZT {z:.1f}".replace(".0", "")
            return (gt.fmt12(ts, 0, date=False, seconds=False),
                    f"day {day} {gt.fmt12(ts, 0, date=False)}{zt}")
        return f
    name = ("after first pellet" if "first pellet" in alignment else
            "from the event" if "event" in alignment else "into recording")

    def f(x):
        return _elapsed(x), f"{_elapsed_long(x)} {name}"
    return f


def ticks(alignment, x0, x1, ref_clock, lights_on=None):
    step = _nice_step(x1 - x0)
    if alignment.startswith("Clock"):
        vals, labs = gt.time_ticks(x0, x1, ref_clock)
        return vals, labs
    lab = axis_labeler(alignment, ref_clock, lights_on)
    start = np.ceil(x0 / step) * step
    vals = np.arange(start, x1 + 1e-9, step)
    out = []
    for v in vals:
        short = _elapsed(v, step) if not alignment.startswith("Time of day") else lab(v)[0]
        if alignment.startswith("Time of day"):
            if v % 86400 == 0 or v == vals[0]:
                short = f"Day {int(v // 86400) + 1}<br>{short}"
            if lights_on is not None:
                z = ((v / 3600.0) - lights_on) % 24
                short += f"<br><span style='color:#64748b'>ZT {z:g}</span>"
        out.append(short)
    return vals, out


def axis_title(alignment, event_label=None):
    return {"Clock": "Clock time", "Time of day": "Time of day (recordings from different days overlaid)",
            "Time since recording": "Time since each recording started (0 = start)",
            "Time since first pellet": "Time since each animal's first pellet",
            "Time since an event": f"Time from '{event_label or 'event'}' in each recording (0 = event)"}[
        next(k for k in ("Clock", "Time of day", "Time since recording", "Time since first pellet",
                         "Time since an event") if alignment.startswith(k))]


def dark_bands(alignment, x0, x1, ref_clock, on_h, off_h):
    """Dark-phase intervals on the shared axis (clock and time-of-day only)."""
    if on_h is None or off_h is None or on_h == off_h:
        return []
    if alignment.startswith("Clock"):
        base = (pd.Timestamp(ref_clock) - pd.Timestamp(ref_clock).normalize()).total_seconds()
    elif alignment.startswith("Time of day"):
        base = 0.0
    else:
        return []
    out = []
    d0 = int(np.floor((x0 + base) / 86400)) - 1
    d1 = int(np.ceil((x1 + base) / 86400)) + 1
    for d in range(d0, d1 + 1):
        day = d * 86400 - base
        if off_h > on_h:      # dark from off (evening) to next on (morning)
            a, b = day + off_h * 3600, day + (24 + on_h) * 3600
        else:
            a, b = day + off_h * 3600, day + on_h * 3600
        a, b = max(a, x0), min(b, x1)
        if b > a:
            out.append((a, b))
    return out


def _x_hover(fig, labeler):
    """Readable x at the top of the hover box (first trace on every x axis)."""
    done = set()
    for tr in fig.data:
        if tr.type not in ("scatter", "scattergl", "bar") or tr.x is None or tr.visible in (False, "legendonly"):
            continue
        ax = tr.xaxis or "x"
        if ax in done or (tr.type != "bar" and tr.fill not in (None, "none")):
            continue
        x = np.asarray(tr.x, float)
        lab = np.array([labeler(v)[1] if np.isfinite(v) else "" for v in x], dtype=object)
        body = tr.hovertemplate or f"{tr.name}: %{{y:.4g}}<extra></extra>"
        tr.update(customdata=lab, hovertemplate="<b>%{customdata}</b><br>" + body.replace("<extra></extra>", "")
                  + "<extra></extra>")
        done.add(ax)
    if done:
        fig.update_xaxes(unifiedhovertitle=dict(text=" "))


def _style(fig, height):
    fig.update_layout(template="plotly_white", height=height, hovermode="x unified", dragmode="pan",
                      margin=dict(l=70, r=30, t=50, b=70), font=dict(color="#111827"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=12)))
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7")
    return fig


def _apply_axis(fig, S, alignment, ref, lights_on, rows=None):
    x0, x1 = S["e0"], S["e0"] + S["n_bins"] * S["bin_s"]
    vals, labs = ticks(alignment, x0, x1, ref, lights_on)
    kw = dict(tickvals=vals, ticktext=labs, range=[x0, x1])
    if rows:
        for r in rows:
            fig.update_xaxes(row=r, col=1, **kw)
    else:
        fig.update_xaxes(**kw)
    return x0, x1


# =============================================================================
# FIGURES
# =============================================================================

def _event_lines(fig, evs, rows=None, x_span=None):
    """Dashed red line for every (x, label); one label per cluster at the top.
    rows: subplot rows to draw in (None for a single-panel figure)."""
    evs = [(float(x), str(l)) for x, l in evs if x is not None and np.isfinite(x)]
    if not evs:
        return
    for x in sorted({round(x, 3) for x, _ in evs}):
        kw = dict(x=x, line=dict(color=_EV, width=1.1, dash="dash"), layer="above")
        if rows:
            for r in rows:
                fig.add_vline(row=r, col=1, **kw)
        else:
            fig.add_vline(**kw)
    tol = (x_span or (max(x for x, _ in evs) - min(x for x, _ in evs) or 1.0)) * 0.012
    clusters = []
    for x, lab in sorted(evs):
        if clusters and clusters[-1][2] == lab and x - clusters[-1][1] <= tol:
            clusters[-1][1] = x
            clusters[-1][3] += 1
        else:
            clusters.append([x, x, lab, 1])
    level = []
    for a, b, lab, n in clusters:
        y = 1.0 - 0.045 * (sum(1 for xx in level if abs(xx - a) < tol * 8) % 3)
        level.append(a)
        fig.add_annotation(x=(a + b) / 2, y=y, xref="x", yref="paper", showarrow=False, yanchor="bottom",
                           text=f"{lab}" + (f" ×{n}" if n > 1 else ""), font=dict(size=11, color=_EV),
                           bgcolor="rgba(255,255,255,0.75)")

def _band_traces(fig, x, c, lo, hi, n, color, name, row=None, show=True, unit=""):
    kw = dict(row=row, col=1) if row else {}
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        xb = np.r_[x[ok], x[ok][::-1]]
        yb = np.r_[hi[ok], lo[ok][::-1]]
        fig.add_trace(go.Scatter(x=xb, y=yb, fill="toself", fillcolor=_rgba(color, 0.18), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False, name=f"{name} band"), **kw)
    fig.add_trace(go.Scatter(x=x, y=c, mode="lines", name=name, line=dict(color=color, width=2),
                             customdata=n, showlegend=show,
                             hovertemplate=f"{name}: %{{y:.3g}}{unit} (n = %{{customdata}})<extra></extra>"), **kw)


def combined_figure(S, F, stat, alignment, ref, lights_on, darks, show_each, sig_label, fed_label, height,
                    S2=None, evs=None, min_n=1):
    groups = list(dict.fromkeys(S["groups"])) if S else list(dict.fromkeys(F["groups"]))
    colors = {g: (PALETTE[i % len(PALETTE)] if len(groups) > 1 else None) for i, g in enumerate(groups)}
    rows = 2 if (S and F) else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.58, 0.42][:rows],
                        subplot_titles=[t for t, ok in ((f"{sig_label}: {stat}", S), (f"Food: {fed_label}, {stat}", F))
                                        if ok])
    r = 1
    for M_, row, base_color, unit in ((S, 1, _SIG, ""), (F, rows, _FED, "")):
        if not M_:
            continue
        for g in groups:
            sel = [i for i, gg in enumerate(M_["groups"]) if gg == g]
            if not sel:
                continue
            col = colors[g] or base_color
            if show_each:
                for k, i in enumerate(sel):
                    fig.add_trace(go.Scatter(x=M_["x"], y=M_["M"][i], mode="lines",
                                             line=dict(color=_rgba(col, 0.35), width=0.9),
                                             name=f"{M_['animals'][i]}", legendgroup=f"each{row}",
                                             showlegend=False,
                                             hovertemplate=f"{M_['animals'][i]}: %{{y:.3g}}<extra></extra>"),
                                  row=row, col=1)
            c, lo, hi, n = ga.group_stat(M_["M"][sel], stat)
            if M_ is S and min_n > 1:
                few = n < min(min_n, len(sel))
                c, lo, hi = np.where(few, np.nan, c), np.where(few, np.nan, lo), np.where(few, np.nan, hi)
            what = ("465" if S2 is not None else "signal") if M_ is S else "food"
            name = (f"{g} " if len(groups) > 1 else "") + what + f" (n = {len(sel)})"
            _band_traces(fig, M_["x"], c, lo, hi, n, col, name, row=row, show=(M_ is S or len(groups) > 1))
            if M_ is S and S2 is not None:
                sel2 = [i for i, gg in enumerate(S2["groups"]) if gg == g]
                if sel2:
                    c2, lo2, hi2, n2 = ga.group_stat(S2["M"][sel2], stat)
                    if min_n > 1:
                        few = n2 < min(min_n, len(sel2))
                        c2, lo2, hi2 = (np.where(few, np.nan, v_) for v_ in (c2, lo2, hi2))
                    _band_traces(fig, S2["x"], c2, lo2, hi2, n2, _UV,
                                 (f"{g} " if len(groups) > 1 else "") + f"405 (n = {len(sel2)})", row=1)
        r += 1
    ref_S = S or F
    for row in range(1, rows + 1):
        for a, b in darks:
            fig.add_vrect(x0=a, x1=b, fillcolor=_DARK, line_width=0, layer="below", row=row, col=1)
    x0, x1 = _apply_axis(fig, ref_S, alignment, ref, lights_on, rows=list(range(1, rows + 1)))
    _event_lines(fig, evs or [], rows=list(range(1, rows + 1)), x_span=x1 - x0)
    fig.update_xaxes(title_text=axis_title(alignment), row=rows, col=1)
    if S:
        fig.update_yaxes(title_text=sig_label, row=1, col=1)
        fig.add_hline(y=0, line=dict(color="#94a3b8", width=0.8, dash="dot"), row=1, col=1)
    if F:
        fig.update_yaxes(title_text=fed_label, row=rows, col=1, rangemode="tozero")
    return _style(fig, height)


def each_figure(S, F_counts, members, alignment, ref, lights_on, darks, layout, sig_label, show_bars, height_mult,
                S2=None, row_events=None, own_axis_405=False):
    """row_events: {row index: [(x, label), ...]} on the shared axis."""
    names = S["rows"]
    k = len(names)
    row_events = row_events or {}
    if layout == "One graph":
        fig = go.Figure()
        for i, nm in enumerate(names):
            col = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(x=S["x"], y=S["M"][i], mode="lines", name=S["animals"][i],
                                     line=dict(color=col, width=1.4), legendgroup=str(i),
                                     hovertemplate=f"{S['animals'][i]}: %{{y:.3g}}<extra></extra>"))
            if S2 is not None and i < len(S2["M"]):
                fig.add_trace(go.Scatter(x=S2["x"], y=S2["M"][i], mode="lines", name=f"{S['animals'][i]} 405",
                                         line=dict(color=col, width=1.0, dash="dot"), legendgroup=str(i),
                                         showlegend=False,
                                         hovertemplate=f"{S['animals'][i]} 405: %{{y:.3g}}<extra></extra>"))
        for a, b in darks:
            fig.add_vrect(x0=a, x1=b, fillcolor=_DARK, line_width=0, layer="below")
        x0, x1 = _apply_axis(fig, S, alignment, ref, lights_on)
        _event_lines(fig, [e for v in row_events.values() for e in v], x_span=x1 - x0)
        if S2 is not None:
            fig.add_annotation(x=1, y=-0.16, xref="paper", yref="paper", showarrow=False, xanchor="right",
                               text="solid: 465 · dotted: 405", font=dict(size=11, color="#64748b"))
        fig.update_xaxes(title_text=axis_title(alignment))
        fig.update_yaxes(title_text=sig_label)
        return _style(fig, int(560 * height_mult))
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=min(0.035, 0.25 / max(k, 1)),
                        specs=[[{"secondary_y": True}] for _ in range(k)],
                        subplot_titles=[f"{S['animals'][i]} · {S['groups'][i]}" if len(set(S["groups"])) > 1
                                        else S['rows'][i] if S['animals'][i] == S['rows'][i]
                                        else f"{S['animals'][i]}  ·  {S['rows'][i]}" for i in range(k)])
    by_name = {m["name"]: m for m in members}
    for i in range(k):
        row = i + 1
        y = S["M"][i]
        fig.add_trace(go.Scatter(x=S["x"], y=y, mode="lines", name="465" if S2 is not None else S["animals"][i],
                                 showlegend=bool(S2 is not None and i == 0), legendgroup="465",
                                 line=dict(color=_SIG, width=1.2),
                                 hovertemplate=f"{S['animals'][i]}" + (" 465" if S2 is not None else "")
                                 + ": %{y:.4g}<extra></extra>"),
                      row=row, col=1, secondary_y=False)
        if S2 is not None and i < len(S2["M"]):
            fig.add_trace(go.Scatter(x=S2["x"], y=S2["M"][i], mode="lines", name="405", showlegend=(i == 0),
                                     legendgroup="405", line=dict(color=_UV, width=1.1),
                                     hovertemplate=f"{S['animals'][i]} 405: %{{y:.4g}}<extra></extra>"),
                          row=row, col=1, secondary_y=own_axis_405)
            if own_axis_405:
                fig.update_yaxes(secondary_y=True, row=row, col=1, showgrid=False, nticks=3,
                                 title_text="405" if i == k // 2 else None, title_font=dict(color=_UV),
                                 tickfont=dict(color=_UV))
        if show_bars and F_counts is not None and not own_axis_405:
            j = next((jj for jj, a in enumerate(F_counts["animals"]) if a == S["animals"][i]
                      and F_counts["groups"][jj] == S["groups"][i]), None)
            if j is not None:
                fig.add_trace(go.Bar(x=F_counts["x"], y=F_counts["M"][j], marker_color=_rgba(_FED, 0.55),
                                     name="pellets", showlegend=False,
                                     hovertemplate="%{y:.0f} pellets<extra></extra>"),
                              row=row, col=1, secondary_y=True)
                fig.update_yaxes(secondary_y=True, row=row, col=1, showgrid=False, rangemode="tozero",
                                 nticks=3, tickformat=".0f", title_text="pellets" if i == k // 2 else None)
        m = by_name.get(S["rows"][i].split(" (mean of")[0])
        if m is not None and m["pellets"].size:
            off = ga.align_offset(m, alignment, ref)
            if off is not None:
                a, b = ga.window_of(m)
                p = m["pellets"][(m["pellets"] >= a) & (m["pellets"] <= b)] + off
                top = np.nanpercentile(y, 99) if np.isfinite(y).any() else 1
                fig.add_trace(go.Scatter(x=p, y=np.full(p.size, top), mode="markers", showlegend=False,
                                         marker=dict(symbol="line-ns-open", size=7, color=_FED, line=dict(width=1)),
                                         hoverinfo="skip", name="pellet"), row=row, col=1, secondary_y=False)
        for a, b in darks:
            fig.add_vrect(x0=a, x1=b, fillcolor=_DARK, line_width=0, layer="below", row=row, col=1)
        for x, lab in row_events.get(i, []):
            fig.add_vline(x=x, line=dict(color=_EV, width=1.1, dash="dash"), row=row, col=1)
    x0, x1 = _apply_axis(fig, S, alignment, ref, lights_on, rows=list(range(1, k + 1)))
    allev = [e for v in row_events.values() for e in v]
    if allev:   # labels once, at the top
        tmp = go.Figure()
        _event_lines(tmp, allev, x_span=x1 - x0)
        for a in tmp.layout.annotations:
            fig.add_annotation(a)
    fig.update_xaxes(title_text=axis_title(alignment), row=k, col=1)
    fig.update_yaxes(title_text=sig_label, row=(k + 1) // 2, col=1, secondary_y=False)
    fig.update_annotations(font=dict(size=12, color="#334155"))
    return _style(fig, int((150 * k + 140) * height_mult))


def heatmap_figure(S, F, alignment, ref, lights_on, sig_label, centred, height_mult, row_events=None):
    rows = 2 if F is not None else 1
    labels = [f"{a} · {g}" if len(set(S["groups"])) > 1 else a for a, g in zip(S["animals"], S["groups"])]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[sig_label, "Pellets per bin"][:rows])
    zmax = np.nanpercentile(np.abs(S["M"]), 98) if np.isfinite(S["M"]).any() else 1
    fig.add_trace(go.Heatmap(x=S["x"], y=labels, z=S["M"], colorscale="RdBu_r" if centred else "Viridis",
                             zmid=0 if centred else None, zmin=-zmax if centred else None,
                             zmax=zmax if centred else None,
                             colorbar=dict(title=dict(text=sig_label, side="right"), len=0.55 if rows > 1 else 1,
                                           y=1, yanchor="top", thickness=12),
                             hovertemplate="%{y}<br>%{z:.3g}<extra></extra>"), row=1, col=1)
    ex, ey, et = [], [], []
    for i, evs in (row_events or {}).items():
        for x, lab in evs:
            ex.append(x)
            ey.append(labels[i])
            et.append(lab)
    if ex:
        fig.add_trace(go.Scatter(x=ex, y=ey, mode="markers", marker=dict(symbol="line-ns", size=18,
                                 line=dict(color="#111827", width=2.5)), text=et, name="events",
                                 showlegend=False, hovertemplate="%{y}: %{text}<extra></extra>"), row=1, col=1)
    if F is not None:
        fl = [f"{a} · {g}" if len(set(F["groups"])) > 1 else a for a, g in zip(F["animals"], F["groups"])]
        fig.add_trace(go.Heatmap(x=F["x"], y=fl, z=F["M"], colorscale=[[0, "#fff7ed"], [1, "#9a3412"]],
                                 colorbar=dict(title=dict(text="pellets", side="right"), len=0.4, y=0,
                                               yanchor="bottom", thickness=12),
                                 hovertemplate="%{y}<br>%{z:.0f} pellets<extra></extra>"), row=2, col=1)
    _apply_axis(fig, S, alignment, ref, lights_on, rows=list(range(1, rows + 1)))
    fig.update_xaxes(title_text=axis_title(alignment), row=rows, col=1)
    fig.update_yaxes(autorange="reversed")
    n = len(labels)
    fig.update_layout(template="plotly_white", height=int((140 + 34 * n * rows + 120) * height_mult),
                      margin=dict(l=110, r=40, t=50, b=70), font=dict(color="#111827"))
    return fig


def aligned_group_figure(G, R, base, resp, show_each, unit_label, height):
    fig = go.Figure()
    gs = list(G)
    for gi, g in enumerate(gs):
        col = PALETTE[gi % len(PALETTE)] if len(gs) > 1 else _SIG
        if show_each:
            for r in R:
                if r is None or r["group"] != g or r["n_meals"] == 0:
                    continue
                fig.add_trace(go.Scatter(x=r["tax"], y=r["curve"], mode="lines", showlegend=False,
                                         line=dict(color=_rgba(col, 0.35), width=0.9), name=r["animal"],
                                         hovertemplate=f"{r['animal']} ({r['n_meals']} meals): %{{y:.3g}}"
                                                       "<extra></extra>"))
        v = G[g]
        name = (f"{g}: " if len(gs) > 1 else "") + f"mean of {v['n_animals']} animals"
        _band_traces(fig, v["tax"], v["centre"], v["lo"], v["hi"], v["n"], col, name)
    fig.add_vrect(x0=base[0], x1=base[1], fillcolor="rgba(100,116,139,0.10)", line_width=0, layer="below")
    fig.add_vrect(x0=resp[0], x1=resp[1], fillcolor="rgba(37,99,235,0.08)", line_width=0, layer="below")
    fig.add_vline(x=0, line=dict(color="#b45309", width=1.2))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=0.8, dash="dot"))
    fig.add_annotation(x=0, y=1.0, yref="paper", text="meal onset", showarrow=False, yanchor="bottom",
                       font=dict(size=11, color="#b45309"))
    fig.update_xaxes(title_text="Seconds from meal onset")
    fig.update_yaxes(title_text=unit_label)
    return _style(fig, height)


# =============================================================================
# PAGE
# =============================================================================

def _default_bin(span_s):
    """Bin width (s) giving roughly 600-900 bins over the span."""
    want = max(span_s, 1.0) / 800.0
    return min(BIN_OPTS, key=lambda b: abs(np.log(b / want)))


def _bin_label(b):
    return f"{b:g} s" if b < 60 else f"{b / 60:g} min" if b < 3600 else f"{b / 3600:g} h"


# =============================================================================
# EVENTS
# =============================================================================

def parse_recording_time(text):
    """'75', '75.5', '1:15', '0:01:15', '1m15s' -> seconds (or None)."""
    s = str(text or "").strip().lower().replace(" ", "")
    if not s:
        return None
    neg = s.startswith(("-", "−"))
    s = s.lstrip("-−")
    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            v = 0.0
            for p in parts:
                v = v * 60 + p
        elif any(c in s for c in "hms"):
            import re
            v = 0.0
            for num, unit in re.findall(r"([\d.]+)([hms])", s):
                v += float(num) * {"h": 3600, "m": 60, "s": 1}[unit]
        else:
            v = float(s)
    except ValueError:
        return None
    return -v if neg else v


def parse_clock_time(text, clock_start):
    """'8:25:30 PM', '20:25:30', '2026-11-08 8:25 PM' -> seconds since clock_start (or None).
    Without a date, the date of the recording is used (next day if that puts it more than
    an hour before the start, e.g. a recording that runs past midnight)."""
    if clock_start is None:
        return None
    s = str(text or "").strip()
    if not s:
        return None
    try:
        ts = pd.Timestamp(s)
    except (ValueError, TypeError):
        return None
    has_date = any(c in s for c in "-/") or len(s) > 12 and any(ch.isalpha() for ch in s.replace("AM", "")
                                                                 .replace("PM", "").replace("am", "")
                                                                 .replace("pm", ""))
    cs = pd.Timestamp(clock_start)
    if not has_date:
        ts = cs.normalize() + (ts - ts.normalize())
        if (ts - cs).total_seconds() < -3600:
            ts += pd.Timedelta(days=1)
    return (ts - cs).total_seconds()


def resolve_events(ev_df, recordings, clock_starts):
    """Event table -> {recording: [(seconds since its start, label), ...]} and a list of problems."""
    out = {r: [] for r in recordings}
    bad = []
    if ev_df is None or ev_df.empty:
        return out, bad
    for i, row in ev_df.iterrows():
        label = str(row.get("event") or "").strip() or "Event"
        target = str(row.get("recording") or "All recordings")
        when = row.get("time")
        kind = str(row.get("time is") or EVENT_TIME_KINDS[0])
        if when is None or str(when).strip() in ("", "nan", "None"):
            continue
        targets = recordings if target.startswith("All") else [target] if target in out else []
        for r in targets:
            if kind.startswith("Clock"):
                sec = parse_clock_time(when, clock_starts.get(r))
            else:
                sec = parse_recording_time(when)
            if sec is None:
                bad.append(f"row {i + 1} ('{label}', {when}): time not understood"
                           + (" (this recording has no clock time)" if kind.startswith("Clock")
                              and clock_starts.get(r) is None else ""))
                break
            out[r].append((float(sec), label))
    return out, bad


def _events_editor(names, clock_starts):
    """Event table (any number of events, each for all or one recording) with a quick-add row."""
    ss = st.session_state
    cols = ["event", "recording", "time", "time is"]
    if "grp_events_base" not in ss:
        ss["grp_events_base"] = pd.DataFrame([], columns=cols)
        ss["grp_events_ver"] = 0
    with st.expander("Events", expanded=not len(ss["grp_events_base"])):
        th.explain("How events work",
                   "Mark moments in the recordings, e.g. food going in or an injection.\n\n"
                   "* **Recording time**: `60`, `1:00` or `0:01:00` all mean one minute after that recording "
                   "started.\n* **Clock time**: the time of day, e.g. `8:25:30 PM`; each recording places it on "
                   "its own timeline.\n* One row can apply to every recording or to one.\n\n"
                   "Events show as dashed red lines in every group graph, can be used to line recordings up "
                   "(*Line recordings up by → Time since an event*), are graded in *Event response & signal "
                   "check*, and the AI can answer questions about them.")
        q = st.columns([1.3, 1.0, 1.1, 1.0])
        lab = q[0].text_input("Event", "Food in", key="grp_q_label")
        when = q[1].text_input("Time", "1:00", key="grp_q_time")
        kind = q[2].selectbox("Time is", EVENT_TIME_KINDS, key="grp_q_kind")
        q[3].write("")
        if q[3].button("Add to every recording", key="grp_q_add", use_container_width=True):
            cur = ss.get("grp_events_cur", ss["grp_events_base"])
            ss["grp_events_base"] = pd.concat([cur, pd.DataFrame([{"event": lab, "recording": "All recordings",
                                                                   "time": when, "time is": kind}])],
                                              ignore_index=True)
            ss["grp_events_ver"] += 1
            st.rerun()
        ed = st.data_editor(ss["grp_events_base"], key=f"grp_events_{ss['grp_events_ver']}", num_rows="dynamic",
                            hide_index=True, use_container_width=True,
                            column_config={
                                "event": st.column_config.TextColumn("Event", required=True),
                                "recording": st.column_config.SelectboxColumn(
                                    "Recording", options=["All recordings"] + list(names), required=True,
                                    default="All recordings"),
                                "time": st.column_config.TextColumn("Time", help="60, 1:00, 0:01:00 or 8:25:30 PM"),
                                "time is": st.column_config.SelectboxColumn("Time is", options=EVENT_TIME_KINDS,
                                                                            default=EVENT_TIME_KINDS[0]),
                            })
        ss["grp_events_cur"] = ed
        ev, bad = resolve_events(ed, list(names), clock_starts)
        for b in bad:
            st.markdown(f'<div class="pp-note warn">Event {b}</div>', unsafe_allow_html=True)
        if any(ev.values()):
            th.explain("Where each event landed", "\n".join(
                f"* **{r}**: " + ", ".join(f"{l} at {gt.fmt_hms(s_, 1)}" + (
                    f" ({gt.fmt12(pd.Timestamp(clock_starts[r]) + pd.Timedelta(seconds=s_), 0, date=False)})"
                    if clock_starts.get(r) is not None else "") for s_, l in v)
                for r, v in ev.items() if v))
    ss["grp_events"] = ev
    return ev


def _row_events(S, members, alignment, ref, label=None):
    """{row index of S: [(x on the shared axis, label)]}; collapsed animal rows get all their recordings' events."""
    out = {}
    by_name = {m["name"]: m for m in members}
    for i, rname in enumerate(S["rows"]):
        if rname in by_name:
            mm = [by_name[rname]]
        else:
            mm = [m for m in members if (m["animal"] or m["name"]) == S["animals"][i] and m["group"] == S["groups"][i]]
        evs = []
        for m in mm:
            off = ga.align_offset(m, alignment, ref)
            if off is None:
                continue
            evs += [(sec + off, lab) for sec, lab in m.get("events", []) if label is None or lab == label]
        if evs:
            out[i] = evs
    return out


# =============================================================================
# PAGE
# =============================================================================

def _series(df, roi, kind):
    """-> t, y, unit, y405 (or None). Raw kinds use the 10 Hz-filtered channels when present."""
    cols = df.columns

    def col(*names):
        return next((c for c in names if c in cols), None)
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    if kind in (fed_view.RAW_BOTH, RATIO):
        c465 = col(f"{roi}_sig_raw", f"{roi}_sig_filt", f"{roi}_sig_filt_10Hz_lowpass")
        c405 = col(f"{roi}_uv_raw", f"{roi}_uv_filt", f"{roi}_uv_filt_10Hz_lowpass")
        if not (c465 and c405):
            raise KeyError("raw 465 and 405 columns")
        y = pd.to_numeric(df[c465], errors="coerce").to_numpy(float)
        u = pd.to_numeric(df[c405], errors="coerce").to_numpy(float)
        if kind == RATIO:
            with np.errstate(divide="ignore", invalid="ignore"):
                return t, y / u, "465/405", None
        return t, y, "V", u
    t, y, unit, _ = gt._panel_series(df, roi, kind, 0.0)
    return t, y, unit, None


def _raw_for_check(df, roi):
    """t, 465, 405, dF/F % for the signal check (filtered channels preferred)."""
    c = df.columns

    def col(*names):
        return next((x for x in names if x in c), None)
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    c465 = col(f"{roi}_sig_filt", f"{roi}_sig_filt_10Hz_lowpass", f"{roi}_sig_raw")
    c405 = col(f"{roi}_uv_filt", f"{roi}_uv_filt_10Hz_lowpass", f"{roi}_uv_raw")
    sig = pd.to_numeric(df[c465], errors="coerce").to_numpy(float) if c465 else None
    iso = pd.to_numeric(df[c405], errors="coerce").to_numpy(float) if c405 else None
    dff = None
    try:
        _, dff, _, _ = gt._panel_series(df, roi, next(k for k in gt.panel_types(df, roi) if k.startswith("dF/F")),
                                        0.0)
    except (StopIteration, KeyError):
        pass
    return t, sig, iso, dff


def render(results, fed, roi, prepare_df, max_points=12_000, height_mult=1.0, config=None):
    if len(results) < 1:
        st.info("Load two or more recordings to analyse them together.")
        return
    ss = st.session_state
    animals = list(fed.animals) if fed is not None and not fed.events.empty else []

    th.explain("About this view",
               "Analyse several recordings together: the group mean with its spread, every recording on the same "
               "time axis, heatmaps, responses to events you mark (with a grade for each recording), and the meal "
               "response with each animal as one data point.\n\n" + REFERENCES)

    # ---- members --------------------------------------------------------
    rows = []
    for i, r in enumerate(results):
        sid = (r.get("settings") or {}).get("subject_ID", "")
        guess = fed3.guess_animal(r["name"], animals, sid) if animals else None
        short = str(r["name"]).split("_")[0].split("-")[0] or str(r["name"])
        rows.append({"include": True, "recording": str(r["name"]),
                     "animal": guess or fed3.animal_from_name(r["name"]) or short, "group": "All"})
    base_df = pd.DataFrame(rows)
    with st.expander("Recordings, animals and groups", expanded=len(results) <= 8):
        st.caption("Untick a recording to leave it out. Give recordings a **group** label to compare conditions "
                   "or days (e.g. Day 1 / Day 7, or Control / Drug). The same animal in two groups is compared "
                   "as a paired design.")
        opts = sorted(set(animals) | set(base_df["animal"]))
        ed = st.data_editor(base_df, key="grp_editor", hide_index=True, use_container_width=True,
                            disabled=["recording"],
                            column_config={
                                "include": st.column_config.CheckboxColumn("Use", width="small"),
                                "recording": st.column_config.TextColumn("Recording"),
                                "animal": st.column_config.SelectboxColumn("Animal (FED ID)", options=opts),
                                "group": st.column_config.TextColumn("Group"),
                            })
    ed = ed[ed["include"]]
    if ed.empty:
        st.warning("No recording selected.")
        return
    ss["grp_groups"] = dict(zip(ed["recording"], ed["group"].fillna("All").astype(str).str.strip()
                                .replace("", "All")))
    by_name = {str(r["name"]): r for r in results}
    chosen = [by_name[n] for n in ed["recording"]]
    clock_starts = {str(r["name"]): gt.clock_start_of(r["df"], r.get("settings")) for r in chosen}
    durations = [float(pd.to_numeric(r["df"]["time_sec"], errors="coerce").max()
                       - pd.to_numeric(r["df"]["time_sec"], errors="coerce").min()) for r in chosen]

    # ---- events ---------------------------------------------------------
    events = _events_editor(list(ed["recording"]), clock_starts)
    ev_labels = list(dict.fromkeys(l for v in events.values() for _, l in v))

    # ---- settings -------------------------------------------------------
    first = chosen[0]
    kinds = [k for k in fed_view.signal_kinds(first["df"], roi) if "running mean" not in k]
    if fed_view.RAW_BOTH in kinds:
        kinds.insert(kinds.index(fed_view.RAW_BOTH) + 1, RATIO)
    cs_known = [v for v in clock_starts.values() if v is not None]
    spread = (max(cs_known) - min(cs_known)).total_seconds() if len(cs_known) >= 2 else 0.0
    med_dur = float(np.median(durations)) if durations else 0.0
    starts_differ = spread > max(60.0, 0.05 * med_dur)
    c = st.columns([1.5, 1.3, 1.6, 1.2])
    alignment = c[0].selectbox("Line recordings up by", ga.ALIGNMENTS, index=2 if starts_differ else 0,
                               key="grp_align",
                               help="Clock time: same moment (animals recorded together). Time of day: recordings "
                                    "from different days overlaid by time of day (longitudinal). Time since start: "
                                    "every recording starts at 0. Time since the first pellet or an event: each "
                                    "recording's own zero.")
    kind = c[1].selectbox("Signal", kinds, index=next((i for i, k in enumerate(kinds) if k.startswith("dF/F")), 0),
                          key="grp_kind",
                          help="dF/F and z-score are motion-corrected. 'Raw 465 + 405' shows both raw channels; "
                               "the 465 ÷ 405 ratio cancels bleaching and light-path changes common to both.")
    is_raw = kind in RAW_KINDS
    norm_opts = ga.NORMS
    norm = c[2].selectbox("Normalise each recording", norm_opts,
                          index=norm_opts.index("% change from baseline mean") if is_raw else 0,
                          key=f"grp_norm_{'raw' if is_raw else 'proc'}",
                          help="Raw channels are in volts that depend on LED power and fibre, so compare them as "
                               "% change from a baseline.")
    stat = c[3].selectbox("Group line and band", ga.STATS, key="grp_stat")
    ev_label = None
    if alignment.startswith("Time since an event"):
        if not ev_labels:
            st.markdown('<div class="pp-note warn">Add an event above (Events) to line recordings up on it. '
                        'Showing time since recording start meanwhile.</div>', unsafe_allow_html=True)
            alignment = ga.ALIGNMENTS[2]
        else:
            ev_label = st.selectbox("Event to line up on (0 = its first occurrence in each recording)", ev_labels,
                                    key="grp_align_event")
    if starts_differ and alignment.startswith("Clock"):
        th.explain("Why don't the recordings overlap?",
                   f"These recordings started at different clock times ({gt.fmt12(min(cs_known), 0, date=False)} "
                   f"to {gt.fmt12(max(cs_known), 0, date=False)}), so on a clock axis they sit side by side. "
                   "Choose **Time since recording start** to put every start at 0.")
    view = st.radio("Group view", VIEWS, horizontal=True, key="grp_view")

    span_guess = max(durations) if durations else 3600.0
    with st.expander("More settings", expanded=False):
        d = st.columns(4)
        sessions_all = fed.sessions["session"].tolist() if animals and not fed.sessions.empty else []
        sessions = d[0].multiselect("FED sessions", sessions_all,
                                    default=[s for s in sessions_all if not str(s).lower().startswith("test")],
                                    key="grp_sessions") if sessions_all else None
        mg = d[1].number_input("Pellet mass (mg)", 1.0, 1000.0, float(ss.get("fed_mg", 20.0)), 1.0, key="grp_mg")
        preset = d[2].selectbox("Meal definition", list(MEAL_PRESETS), key="grp_meal")
        if MEAL_PRESETS[preset] is None:
            gap = d[3].number_input("Max gap (s)", 5.0, 3600.0, 120.0, 5.0, key="grp_gap")
            minp = d[3].number_input("Min pellets", 1, 50, 2, 1, key="grp_minp")
        else:
            gap, minp = MEAL_PRESETS[preset]
            d[3].caption(f"Meals: pellets less than {gap:g} s apart, at least {minp} pellet(s).")
        e = st.columns(4)
        lights_on = e[0].number_input("Lights on (hour, 0-23)", 0, 23, 7, 1, key="grp_on")
        lights_off = e[1].number_input("Lights off (hour)", 0, 23, 19, 1, key="grp_off")
        shade = e[2].checkbox("Shade the dark phase", True, key="grp_shade")
        dflt_base = 30.0 if span_guess >= 2 * 3600 else 0.5
        base_min = e[3].number_input("Baseline period (min)", 0.1, 600.0, dflt_base, 0.1,
                                     key=f"grp_basemin_{dflt_base:g}",
                                     help="For the baseline normalisations: the first minutes of each recording, "
                                          "or the minutes before the first pellet / the event when lining up on "
                                          "it.")
        f = st.columns(4)
        per_animal = f[0].checkbox("Average recordings of the same animal first", True, key="grp_peranimal",
                                   help="So each animal counts once (Loewinger et al. 2024).")
        fed_only = f[1].checkbox("Add FED-only animals to the food plots", False, key="grp_fedonly",
                                 help="Animals with FED data but no recording (clock / time-of-day alignment).")
        pre_post = f[2].text_input("Meal window: before, after (s)", "300, 600", key="grp_prepost")
        bw = f[3].text_input("Baseline, response (s)", "-100,-50 / 0,50", key="grp_windows")

    try:
        pre, post = (float(v) for v in pre_post.split(","))
        b_s, r_s = bw.split("/")
        base = tuple(float(v) for v in b_s.split(","))
        resp = tuple(float(v) for v in r_s.split(","))
        assert pre > 0 and post > 0 and base[0] < base[1] and resp[0] < resp[1]
    except Exception:
        st.warning("Meal windows not understood; using 300, 600 and -100,-50 / 0,50.")
        pre, post, base, resp = 300.0, 600.0, (-100.0, -50.0), (0.0, 50.0)

    if view == "Event response & signal check":
        _signal_check_view(ed, by_name, events, ev_labels, roi, prepare_df, config, height_mult, clock_starts)
        return

    # ---- build members --------------------------------------------------
    members, problems = [], []
    for _, row in ed.iterrows():
        r = by_name[row["recording"]]
        df = prepare_df(r["df"])
        if df is None or df.empty:
            problems.append(f"{row['recording']}: no data in the visual window")
            continue
        try:
            t, y, unit, y405 = _series(df, roi, kind)
        except Exception as ex:
            problems.append(f"{row['recording']}: {kind} not available ({ex})")
            continue
        cs = clock_starts.get(row["recording"])
        try:
            ctrl = fed_view._control_405(r["df"], roi)
            ctrl = ctrl if ctrl[0] is not None else None
        except Exception:
            ctrl = None
        an = row["animal"] if row["animal"] not in (None, "", "(none)") else None
        m = ga.make_member(row["recording"], an, ss["grp_groups"][row["recording"]], t, y, cs,
                           fed if animals else None, sessions,
                           (r.get("settings") or {}).get("fit_R2"), ctrl, mg / 1000.0)
        m["y405"] = y405
        m["events"] = events.get(row["recording"], [])
        if ev_label:
            hits = [sec for sec, l in m["events"] if l == ev_label]
            m["event_t"] = min(hits) if hits else None
        members.append(m)
    for p in problems:
        st.markdown(f'<div class="pp-note warn">{p}</div>', unsafe_allow_html=True)
    if not members:
        return
    unit = _series(prepare_df(first["df"]), roi, kind)[2]
    norm_short = {"z-score (whole recording)": "z", "Robust z (median / MAD)": "robust z",
                  "z-score to a baseline period": "z vs baseline", "Subtract baseline mean": f"Δ {unit}".strip(),
                  "% change from baseline mean": "% change", "As recorded": unit,
                  "Scale 0–1 (1st–99th percentile)": "0–1 scaled"}[norm]
    kname = "465 raw" if kind == fed_view.RAW_BOTH else kind.split(" (")[0]
    sig_label = f"{kname} ({norm_short})" if norm_short else kname
    centred = norm.startswith(("z", "Robust", "Subtract", "%"))

    fed_members = list(members)
    if fed_only and animals and alignment.startswith(("Clock", "Time of day")):
        have = {m["animal"] for m in members}
        dflt = ed["group"].mode().iat[0] if not ed["group"].mode().empty else "All"
        for an in animals:
            if an in have:
                continue
            cov = fed.coverage_of(an, sessions)
            if len(cov):
                fed_members.append(ga.make_member(f"FED {an}", an, dflt, None, None, cov["start"].min(), fed,
                                                  sessions, None, None, mg / 1000.0))
    ref = ga.reference_clock(fed_members)
    spans = []
    for m in members:
        off = ga.align_offset(m, alignment, ref)
        if off is not None:
            a, b = ga.window_of(m)
            spans += [a + off, b + off]
    span = (max(spans) - min(spans)) if spans else 3600
    g1 = st.columns([1.3, 1.0, 1.0, 1.4, 1.3])
    dflt_bin = _default_bin(span)
    bin_s = g1[0].select_slider("Bin", BIN_OPTS, value=dflt_bin, format_func=_bin_label,
                                key=f"grp_binsec_{dflt_bin:g}",
                                help="Each recording is averaged in bins of this width before the group "
                                     "statistics; the default gives ~800 points across the graph.")
    rec_on, rec_w, why = sc.recommend_smoothing(max(durations) if durations else span)
    sm_on = g1[1].toggle("Sliding mean", value=rec_on, key=f"grp_sm_on_{int(rec_on)}",
                         help=sc.SMOOTHING_HELP)
    if max(durations or [span]) < 2 * 3600:
        sm_min = g1[2].number_input("Window (s)", 0.5, 7200.0, float(rec_w), 1.0, key=f"grp_sm_ws_{rec_w:g}",
                                    disabled=not sm_on, help="Recommended for this recording length: " + why) / 60.0
    else:
        sm_min = g1[2].number_input("Window (min)", 0.05, 240.0, float(round(rec_w / 60.0, 2)), 0.25,
                                    key=f"grp_sm_w_{rec_w:g}", disabled=not sm_on,
                                    help="Recommended for this recording length: " + why)
    fed_mode = g1[3].selectbox("Food shown as", ga.FED_MODES, key="grp_fedmode") \
        if animals and view == "Combined (group mean)" else "Pellets per bin"
    show_each = g1[4].checkbox("Show each animal faintly", True, key="grp_each") \
        if view in ("Combined (group mean)", "Meal-aligned (group)") else False
    min_n = 1
    if view == "Combined (group mean)" and len(members) >= 3:
        min_n = g1[4].number_input("Hide the mean where fewer animals have data than", 1, len(members),
                                   min(3, len(members)), 1, key="grp_min_n",
                                   help="Recordings of different lengths: at the end only one or two animals are "
                                        "left, and their 'mean' is not a group mean.")
    k = max(1, int(round(sm_min * 60.0 / bin_s))) if sm_on else 1
    if sm_on and k <= 1:
        st.caption(f"The sliding mean ({sm_min:g} min) is not longer than one bin ({_bin_label(bin_s)}), so it "
                   "changes nothing; use a smaller bin or a longer window.")
    notes = [ga.METHOD_NOTES.get(norm, ""), ga.METHOD_NOTES.get(stat, "")]
    if view == "Combined (group mean)" and animals:
        notes.append(ga.METHOD_NOTES.get(fed_mode, ""))
    th.explain("What these settings do",
               f"**Sliding mean** ({'on' if sm_on else 'off'}). Recommended for these recordings: {why}\n\n"
               + sc.SMOOTHING_HELP + "\n\n" + f"**{norm}.** {notes[0]}\n\n**{stat}.** {notes[1]}"
               + (f"\n\n**{fed_mode}.** {notes[2]}" if len(notes) > 2 and notes[2] else ""))

    S = ga.signal_matrix(members, alignment, norm, bin_s, k, base_min, ref, per_animal)
    if S is None:
        st.warning("None of the selected recordings can be lined up this way (e.g. no pellets for "
                   "'Time since first pellet', or no event).")
        return
    S2 = None
    if kind == fed_view.RAW_BOTH:
        m405 = [dict(m, y=m["y405"]) for m in members if m.get("y405") is not None]
        S2 = ga.signal_matrix(m405, alignment, norm, bin_s, k, base_min, ref, per_animal) if m405 else None
    if S["skipped"]:
        st.markdown(f'<div class="pp-note warn">Left out (cannot be aligned this way): '
                    f'{", ".join(S["skipped"])}</div>', unsafe_allow_html=True)
    lab = axis_labeler(alignment, ref, lights_on)
    x0, x1 = S["e0"], S["e0"] + S["n_bins"] * S["bin_s"]
    darks = dark_bands(alignment, x0, x1, ref, lights_on, lights_off) if shade else []
    h = height_mult
    clock_for_ruler = ref if alignment.startswith("Clock") else None
    rev = _row_events(S, members, alignment, ref)

    def retitle(fig):
        if ev_label:
            for ax in fig.layout:
                if ax.startswith("xaxis") and fig.layout[ax].title.text:
                    fig.layout[ax].title.text = fig.layout[ax].title.text.replace("'event'", f"'{ev_label}'")
        return fig

    if view == "Combined (group mean)":
        F = ga.fed_matrix(fed_members, alignment, bin_s, fed_mode, ref, per_animal=per_animal) \
            if animals else None
        fed_label = {"Pellets per bin": f"pellets / {_bin_label(bin_s)}", "Cumulative intake (g)": "cumulative g",
                     "Cumulative, % of own total": "% of own total", "Animals eating (%)": "% of animals eating",
                     "Eating rate (pellets/h, rolling)": "pellets / h"}[fed_mode]
        fig = combined_figure(S, F, stat, alignment, ref, lights_on if alignment.startswith("Time of day") else None,
                              darks, show_each, sig_label, fed_label, int(760 * h), S2=S2,
                              evs=[e for v in rev.values() for e in v], min_n=min_n)
        _x_hover(fig, lab)
        gt.chart_block(retitle(fig), key="grp_combined", max_points=max_points, clock_start=clock_for_ruler,
                       config=config)
        _downloads(S, F, stat, lab, "combined", S2)

    elif view == "Each recording":
        layout = st.radio("Layout", ["Stacked panels", "One graph"], horizontal=True, key="grp_layout")
        own_axis = S2 is not None and norm.startswith("As recorded")
        bars = st.checkbox("Pellets per bin behind each trace", True, key="grp_bars") \
            if layout == "Stacked panels" and animals and not own_axis else False
        Fc = ga.fed_matrix(members, alignment, bin_s, "Pellets per bin", ref, per_animal=per_animal) \
            if bars else None
        fig = each_figure(S, Fc, members, alignment, ref, lights_on if alignment.startswith("Time of day") else None,
                          darks, layout, sig_label, bars, h, S2=S2, row_events=rev, own_axis_405=own_axis)
        _x_hover(fig, lab)
        gt.chart_block(retitle(fig), key="grp_each_fig", max_points=max_points * 2, clock_start=clock_for_ruler,
                       config=config)
        th.explain("How to read this graph", "Dashed red lines: events. Ticks above each trace mark single "
                   "pellets; shaded areas are the dark phase." + (" Blue: 465, green: 405." if S2 is not None
                                                                     else ""))
        _downloads(S, None, stat, lab, "each", S2)

    elif view == "Heatmap":
        F = ga.fed_matrix(members, alignment, bin_s, "Pellets per bin", ref, per_animal=per_animal) \
            if animals else None
        fig = heatmap_figure(S, F, alignment, ref, lights_on if alignment.startswith("Time of day") else None,
                             sig_label, centred, h, row_events=rev)
        if S2 is not None:
            th.explain("How to read this graph", "The heatmap shows the 465 channel; see *Each recording* for 405.")
        st.plotly_chart(retitle(fig), use_container_width=True, config=config, key="grp_heat")

    elif view == "Meal-aligned (group)":
        _aligned_view(members, norm, base_min, alignment, gap, minp, pre, post, base, resp, stat, show_each,
                      sig_label, config, h)

    else:
        tab, summ = ga.meal_table(fed_members if animals else members, gap, minp, lights_on, lights_off)
        if tab.empty or not animals:
            st.info("Load FED3 data to see feeding patterns.")
            return
        st.markdown("**Group mean ± SEM**")
        st.dataframe(summ, hide_index=True, use_container_width=True)
        st.markdown("**Each animal**")
        st.dataframe(tab.round(3), hide_index=True, use_container_width=True)
        th.explain("What these columns mean", f"Meals: pellets less than {gap:g} s apart, at least {minp} pellet(s). Satiety ratio: minutes "
                   "without eating after a meal per pellet in that meal. Hours logged excludes times the FED "
                   "was not recording. % pellets in dark uses the lights-off/on hours under More settings.")
        st.download_button("Download feeding pattern (CSV)", tab.to_csv(index=False).encode(),
                           "feeding_pattern_per_animal.csv", "text/csv")


# =============================================================================
# EVENT RESPONSE & SIGNAL CHECK
# =============================================================================

def _pair(text, default):
    try:
        a, b = (float(v) for v in str(text).replace("to", ",").split(","))
        return (a, b) if a < b else default
    except Exception:
        return default


def run_signal_check(recs, roi, event_times, signal=sc.SIGNALS[0], baseline=sc.BASELINE, response=sc.RESPONSE,
                     direction=sc.DIRECTIONS[0], prepare_df=None):
    """recs: [(name, animal, result)], event_times: {name: seconds}. -> list of (name, animal, assessment)."""
    out = []
    for name, animal, r in recs:
        if name not in event_times or event_times[name] is None:
            continue
        df = prepare_df(r["df"]) if prepare_df else r["df"]
        t, sig, iso, dff = _raw_for_check(df, roi)
        if sig is None:
            continue
        a = sc.assess(t, sig, iso, float(event_times[name]), signal=signal, dff=dff, baseline=baseline,
                      response=response, direction=direction)
        a["_t"], a["_sig"], a["_iso"], a["_dff"] = t, sig, iso, dff
        out.append((name, animal, a))
    return out


def _trace_pct(a, which):
    """Event-aligned % change trace of one assessment: 'graded', '465' or '405'."""
    t, sig, iso, dff = a["_t"], a["_sig"], a["_iso"], a["_dff"]
    ev, base = a["event_s"], a["baseline"]
    if which == "465":
        return sc.pct_trace(t, sig, ev, base)
    if which == "405":
        return sc.pct_trace(t, iso, ev, base) if iso is not None else (t - ev, np.full_like(t, np.nan))
    if a["signal"].startswith("465 ÷ 405") and iso is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            return sc.pct_trace(t, sig / iso, ev, base)
    if a["signal"].startswith("dF/F") and dff is not None:
        return sc.diff_trace(t, dff, ev, base)
    return sc.pct_trace(t, sig, ev, base)


def _thin(x, y, dt=0.25):
    """Bin-average to dt for plotting (20 Hz x 13 recordings is a lot of points)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return x, y
    x, y = x[ok], y[ok]
    idx = np.floor((x - x[0]) / dt).astype(int)
    s = np.bincount(idx, y)
    c = np.bincount(idx)
    k = c > 0
    return x[0] + (np.flatnonzero(k) + 0.5) * dt, s[k] / c[k]


def signal_check_figure(A, pre, post, layout, height_mult, smooth_s=1.0):
    base, resp = A[0][2]["baseline"], A[0][2]["response"]
    unit = "% change" if not A[0][2]["signal"].startswith("dF/F") else "Δ dF/F (% points)"
    sig_name = A[0][2]["signal"].split(" (")[0]
    if layout == "Grid: 465 and 405 for each":
        n = len(A)
        ncol = 3 if n > 4 else 2 if n > 1 else 1
        nrow = int(np.ceil(n / ncol))
        fig = make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, shared_yaxes=True,
                            vertical_spacing=min(0.08, 0.5 / nrow), horizontal_spacing=0.04,
                            subplot_titles=[f"{an or nm} · {a['tier']} ({a.get('change', np.nan):+.1f}%)"
                                            if np.isfinite(a.get("change", np.nan)) else f"{an or nm} · {a['tier']}"
                                            for nm, an, a in A])
        for i, (nm, _, a) in enumerate(A):
            r, cc = i // ncol + 1, i % ncol + 1
            for which, col, w in (("465", _SIG, 1.1), ("405", _UV, 1.0)):
                x, y = _trace_pct(a, which)
                k = (x >= -pre) & (x <= post)
                x, y = _thin(x[k], y[k], smooth_s)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=col, width=w), name=which,
                                         legendgroup=which, showlegend=(i == 0),
                                         hovertemplate=f"{nm} {which}: %{{y:.1f}}%<extra></extra>"), row=r, col=cc)
            fig.add_vrect(x0=base[0], x1=base[1], fillcolor="rgba(100,116,139,0.10)", line_width=0, layer="below",
                          row=r, col=cc)
            fig.add_vrect(x0=resp[0], x1=resp[1], fillcolor="rgba(37,99,235,0.07)", line_width=0, layer="below",
                          row=r, col=cc)
            fig.add_vline(x=0, line=dict(color=_EV, width=1.1, dash="dash"), row=r, col=cc)
        fig.update_annotations(font=dict(size=11, color="#334155"))
        fig.update_xaxes(title_text="s from event", row=nrow)
        fig.update_yaxes(title_text="% change", col=1)
        return _style(fig, int((210 * nrow + 120) * height_mult))
    fig = go.Figure()
    if layout == "Group mean ± SEM":
        grid = None
        rows = []
        for nm, _, a in A:
            x, y = _trace_pct(a, "graded")
            g, v = sc.aligned(x, y, 0.0, pre, post, step=max(smooth_s, 0.5))
            rows.append(v)
            grid = g
        M = np.vstack(rows)
        c, lo, hi, n = ga.group_stat(M, "Mean ± SEM")
        _band_traces(fig, grid, c, lo, hi, n, _SIG, f"{sig_name}: mean ± SEM of {len(A)} recordings")
        rows2 = []
        for nm, _, a in A:
            x, y = _trace_pct(a, "405")
            rows2.append(sc.aligned(x, y, 0.0, pre, post, step=max(smooth_s, 0.5))[1])
        c2, lo2, hi2, n2 = ga.group_stat(np.vstack(rows2), "Mean ± SEM")
        _band_traces(fig, grid, c2, lo2, hi2, n2, _UV, "405: mean ± SEM")
    else:
        for nm, an, a in A:
            x, y = _trace_pct(a, "graded")
            k = (x >= -pre) & (x <= post)
            x, y = _thin(x[k], y[k], smooth_s)
            col = sc.TIER_COLOR.get(a["tier"], "#64748b")
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"{an or nm} · {a['tier']} ({a['change']:+.1f}%)"
                                     if np.isfinite(a.get("change", np.nan)) else f"{an or nm} · {a['tier']}",
                                     line=dict(color=col, width=1.3),
                                     hovertemplate=f"{an or nm} ({a['tier']}): %{{y:.1f}}<extra></extra>"))
    fig.add_vrect(x0=base[0], x1=base[1], fillcolor="rgba(100,116,139,0.10)", line_width=0, layer="below")
    fig.add_vrect(x0=resp[0], x1=resp[1], fillcolor="rgba(37,99,235,0.07)", line_width=0, layer="below")
    fig.add_vline(x=0, line=dict(color=_EV, width=1.2, dash="dash"))
    fig.add_hline(y=0, line=dict(color="#94a3b8", width=0.8, dash="dot"))
    for thr in (-10, 10):
        fig.add_hline(y=thr, line=dict(color="#cbd5e1", width=0.8, dash="dot"))
    fig.update_xaxes(title_text="Seconds from the event")
    fig.update_yaxes(title_text=f"{sig_name} ({unit})")
    fig = _style(fig, int(560 * height_mult))
    fig.update_layout(legend=dict(orientation="v", y=1, yanchor="top", x=1.01, xanchor="left"),
                      margin=dict(r=220))
    return fig


def _signal_check_view(ed, by_name, events, ev_labels, roi, prepare_df, config, height_mult, clock_starts):
    ss = st.session_state
    if not ev_labels:
        st.info("Add an event first (Events, above), e.g. 'Food in' at 1:00 in every recording. Each recording "
                "is then graded by how much its signal changes after the event.")
        return
    c = st.columns([1.2, 1.7, 1.5, 1.0, 1.0])
    label = c[0].selectbox("Event", ev_labels, key="sc_event")
    direction = c[1].selectbox("Expected change", sc.DIRECTIONS, key="sc_dir",
                               help="AgRP neurons fall when food appears; most other responses rise.")
    signal = c[2].selectbox("Signal graded", sc.SIGNALS, key="sc_signal",
                            help="465 ÷ 405 of the raw channels (default) cancels bleaching and light-path changes "
                                 "that hit both channels; '465 raw' is the plain % change of 465; dF/F uses the "
                                 "motion-corrected trace.")
    base = _pair(c[3].text_input("Baseline (s)", f"{sc.BASELINE[0]:g}, {sc.BASELINE[1]:g}", key="sc_base"),
                 sc.BASELINE)
    resp = _pair(c[4].text_input("Response (s)", f"{sc.RESPONSE[0]:g}, {sc.RESPONSE[1]:g}", key="sc_resp"),
                 sc.RESPONSE)

    recs = [(n, a, by_name[n]) for n, a in zip(ed["recording"], ed["animal"])]
    first_ev = {n: min((s_ for s_, l in events.get(n, []) if l == label), default=None) for n, _, _ in recs}
    with st.expander("Event time in each recording (edit if one animal got it at a different moment)",
                     expanded=False):
        tdf = pd.DataFrame({"recording": [n for n, _, _ in recs],
                            "event at (s into recording)": [first_ev[n] for n, _, _ in recs]})
        tdf["clock time"] = [gt.fmt12(pd.Timestamp(clock_starts[n]) + pd.Timedelta(seconds=v), 1, date=False)
                             if v is not None and clock_starts.get(n) is not None else ""
                             for n, v in zip(tdf["recording"], tdf["event at (s into recording)"])]
        key = f"sc_times_{abs(hash((label, tuple(first_ev.items())))) % 10 ** 8}"
        tdf = st.data_editor(tdf, key=key, hide_index=True, use_container_width=True,
                             disabled=["recording", "clock time"],
                             column_config={"event at (s into recording)": st.column_config.NumberColumn(
                                 format="%.1f", help="Seconds since that recording started")})
        st.caption("The first occurrence of the event is used. The response often starts a little later than the "
                   "event (the animal has to notice the food); the default response window starts 30 s after it.")
    times = {n: (float(v) if v is not None and np.isfinite(v) else None)
             for n, v in zip(tdf["recording"], tdf["event at (s into recording)"])}

    A = run_signal_check(recs, roi, times, signal, base, resp, direction, prepare_df)
    if not A:
        st.warning("No recording has this event inside it.")
        return
    order = {n: i for i, n in enumerate(sc.TIER_NAMES + ["Not testable"])}
    A.sort(key=lambda x: (order.get(x[2]["tier"], 99), -(x[2].get("score") if np.isfinite(x[2].get("score", np.nan))
                                                         else -1e9)))
    tab = sc.table(A)
    ss["sc_last"] = {"event": label, "signal": signal, "direction": direction, "baseline": base, "response": resp,
                     "rows": tab.drop(columns=["notes"]).to_dict("records")}

    G = sc.group_summary(A)
    chips = " ".join(f'<span style="display:inline-block;padding:2px 10px;margin:2px;border-radius:999px;'
                     f'background:{sc.TIER_COLOR[n]};color:white;font-size:0.85rem">{n}: {k}</span>'
                     for n, k in G["counts"].items() if k)
    good = sum(G["counts"][n] for n in ("Excellent", "Very good", "Good"))
    txt = f"<b>{label}</b>: {len(A)} recordings graded"
    if G.get("n", 0) >= 3:
        txt += f" · mean change {G['mean']:+.1f} % ± {G['sem']:.1f} SEM"
    st.markdown(f'<div class="pp-note">{chips}<br>{txt}</div>', unsafe_allow_html=True)
    stats_md = ""
    if G.get("n", 0) >= 3:
        stats_md = (f"\n\n#### This group\n\n{good} of {len(A)} recordings change by at least 10 %. Across "
                    f"recordings the change is {G['mean']:+.1f} % ± {G['sem']:.1f} SEM (median {G['median']:+.1f} %); "
                    f"one-sample t-test against no change p = {G['t_p']:.2g}"
                    + (f", Wilcoxon p = {G['wilcoxon_p']:.2g}" if np.isfinite(G.get("wilcoxon_p", np.nan)) else "")
                    + ". Each recording counts once.")
    th.explain("Method of classification · why signals look the way they do",
               sc.CRITERIA_MD.format(b0=base[0], b1=base[1], r0=resp[0], r1=resp[1], signal=signal) + stats_md
               + f"\n\n**In the graph:** grey = before ({base[0]:+g} to {base[1]:+g} s), blue = after "
                 f"({resp[0]:+g} to {resp[1]:+g} s), dotted lines at ±10 %.")

    lay = st.radio("Graph", ["Overlay, coloured by grade", "Grid: 465 and 405 for each", "Group mean ± SEM"],
                   horizontal=True, key="sc_layout")
    g = st.columns([1, 1, 1])
    pre = g[0].number_input("Show before (s)", 5.0, 3600.0, float(max(60.0, -base[0] + 10)), 5.0, key="sc_pre")
    post = g[1].number_input("Show after (s)", 10.0, 7200.0, float(max(180.0, resp[1] + 30)), 10.0, key="sc_post")
    sm = g[2].number_input("Plot smoothing (s)", 0.05, 30.0, 1.0, 0.25, key="sc_sm",
                           help="Only for drawing; the grades use the full-resolution data.")
    fig = signal_check_figure(A, pre, post, lay, height_mult, sm)
    gt.chart_block(fig, key="sc_fig", max_points=60_000, clock_start=None, config=config, ruler=False)

    show = tab.copy()
    for col_ in ("change %", "deepest 10 s %", "465 %", "405 %", "noise %", "change ÷ noise", "time to half (s)",
                 "event at (s)", "405 follows 465 (r)"):
        show[col_] = pd.to_numeric(show[col_], errors="coerce").round(2 if "(r)" in col_ else 1)

    def colour(v):
        return f"background-color:{sc.TIER_COLOR.get(v, '#e2e8f0')};color:white;font-weight:600"
    try:
        sty = show.style.map(colour, subset=["grade"]).format(precision=1, na_rep="–").format(
            precision=2, subset=["405 follows 465 (r)"], na_rep="–")
    except AttributeError:
        sty = show.style.applymap(colour, subset=["grade"]).format(precision=1, na_rep="–")
    st.dataframe(sty, hide_index=True, use_container_width=True,
                 column_config={"notes": st.column_config.TextColumn(width="large")})
    dl = st.columns(2)
    dl[0].download_button("Download grades (CSV)", tab.to_csv(index=False).encode(), "signal_check_grades.csv",
                          "text/csv", key="sc_dl1")
    grid = None
    out = {}
    for nm, _, a in A:
        for which in ("graded", "465", "405"):
            x, y = _trace_pct(a, which)
            grid, v = sc.aligned(x, y, 0.0, pre, post, step=0.5)
            out[f"{nm} {which if which != 'graded' else signal.split(' (')[0]}"] = v
    tr = pd.DataFrame({"seconds_from_event": grid, **out})
    dl[1].download_button("Download event-aligned traces (CSV, 0.5 s)", tr.to_csv(index=False).encode(),
                          "signal_check_traces.csv", "text/csv", key="sc_dl2")


def _downloads(S, F, stat, lab, tag, S2=None):
    out = pd.DataFrame({"x_seconds": S["x"], "time": [lab(v)[1] for v in S["x"]]})
    for i, a in enumerate(S["animals"]):
        out[f"signal {a} [{S['groups'][i]}]"] = S["M"][i]
    if S2 is not None and len(S2["x"]) == len(S["x"]):
        for i, a in enumerate(S2["animals"]):
            out[f"405 {a} [{S2['groups'][i]}]"] = S2["M"][i]
    for g in dict.fromkeys(S["groups"]):
        sel = [i for i, gg in enumerate(S["groups"]) if gg == g]
        c, lo, hi, n = ga.group_stat(S["M"][sel], stat)
        out[f"signal {g} centre"], out[f"signal {g} low"], out[f"signal {g} high"], out[f"signal {g} n"] = c, lo, hi, n
    if F is not None and len(F["x"]) == len(S["x"]):
        for i, a in enumerate(F["animals"]):
            out[f"food {a} [{F['groups'][i]}]"] = F["M"][i]
    st.download_button("Download the group data (CSV, one column per animal)", out.to_csv(index=False).encode(),
                       f"group_{tag}.csv", "text/csv", key=f"dl_{tag}")


def _aligned_view(members, norm, base_min, alignment, gap, minp, pre, post, base, resp, stat, show_each,
                  sig_label, config, h):
    with_fed = [m for m in members if m["pellets"].size]
    if not with_fed:
        st.info("No recording has FED pellets matched to it. Set the animal for each recording above.")
        return
    R = []
    with st.spinner("Aligning to meals and testing against random times for every animal…"):
        for m in with_fed:
            mm = dict(m)
            bm = ga.baseline_mask(m, alignment, base_min) if "baseline" in norm.lower() else None
            mm["y"] = ga.normalise(m["t"], m["y"], norm, bm)
            R.append(ga.member_meal_response(mm, gap, minp, pre, post, base, resp, n_perm=1000))
    G = ga.group_meal_response(R, stat)
    if not G:
        st.info("No meals with a full window inside the recordings.")
        return
    fig = aligned_group_figure(G, R, base, resp, show_each, f"{sig_label}, baseline-subtracted", int(520 * h))
    gt.chart_block(fig, key="grp_aligned", max_points=40_000, clock_start=None, config=config, ruler=False)
    th.explain("How to read this graph", f"Grey: baseline {base[0]:g} to {base[1]:g} s; blue: response {resp[0]:g} to {resp[1]:g} s. Each "
               "animal's meals are averaged first, then animals are averaged, so every animal counts once.")

    for g, v in G.items():
        chk = v["check"]
        cls = {"good": "ok", "uncertain": "warn"}.get(chk["level"], "")
        rs = "".join(f"<br>· {r}" for r in chk["reasons"])
        wil = (f"Wilcoxon p = {v['wilcoxon_p']:.3g}" if np.isfinite(v["wilcoxon_p"]) else "Wilcoxon n/a")
        floor = (f" (with {v['n_animals']} animals it cannot go below {v['wilcoxon_min_p']:.3g})"
                 if np.isfinite(v["wilcoxon_min_p"]) and v["n_animals"] <= 6 else "")
        st.markdown(
            f'<div class="pp-note {cls}">🧪 <b>{g}: {chk["label"]}</b><br>{chk["verdict"]}{rs}<br><br>'
            f'Across animals (n = {v["n_animals"]}): mean change {v["mean_delta"]:+.3g} ± {v["sem_delta"]:.2g} SEM; '
            f'one-sample t-test p = {v["t_p"]:.3g}; {wil}{floor}. '
            f'{v["n_sig_animals"]} of {v["n_animals"]} animals change significantly on their own'
            + (f', {v["n_sig_clean"]} of them with a flat 405 control. ' if v["n_sig_clean"] != v["n_sig_animals"]
               else '. ')
            + f'<span style="opacity:.75">Pooling all {v["pooled_meals"]} meals as if independent would give a mean of '
            f'{v["pooled_mean"]:+.3g}; that n is not the number of animals.</span></div>',
            unsafe_allow_html=True)

    gs = list(G)
    if len(gs) >= 2:
        comp = []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                c = ga.compare_groups(R, gs[i], gs[j])
                if c:
                    comp.append({"comparison": f"{gs[j]} − {gs[i]}", "design": c["design"],
                                 "difference": round(c["diff"], 4), "t-test p": round(c["t_p"], 4),
                                 c["nonparam"] + " p": round(c["nonparam_p"], 4) if np.isfinite(c["nonparam_p"])
                                 else None})
        if comp:
            st.markdown("**Between groups** (animal-level)")
            st.dataframe(pd.DataFrame(comp), hide_index=True, use_container_width=True)

    tbl = pd.DataFrame([{
        "animal": r["animal"], "group": r["group"], "meals used": r["n_meals"],
        "mean change": round(r["mean_delta"], 4) if np.isfinite(r["mean_delta"]) else None,
        "p vs random times": round(r["p_vs_random"], 4) if np.isfinite(r["p_vs_random"]) else None,
        "405 change": round(r["control"]["effect"], 3) if r["control"]["effect"] is not None else None,
        "405 p": round(r["control"]["p"], 4) if r["control"]["p"] is not None else None,
        "data check": r["check"]["label"]} for r in R if r is not None])
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    per_meal = []
    for r in R:
        if r is None:
            continue
        for tm, dlt in zip(r["meal_times"], r["deltas_all"]):
            per_meal.append({"animal": r["animal"], "group": r["group"], "recording": r["name"],
                             "meal_onset_sec_into_recording": round(float(tm), 2), "change": float(dlt)})
    st.download_button("Download per-meal changes (CSV, for a mixed model with animal as random effect)",
                       pd.DataFrame(per_meal).to_csv(index=False).encode(), "meal_changes_per_meal.csv", "text/csv")
