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

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#882255"]
_SIG = "#0072B2"
_FED = "#b45309"
_DARK = "rgba(15,23,42,0.055)"

VIEWS = ["Combined (group mean)", "Each recording", "Heatmap", "Meal-aligned (group)", "Feeding pattern table"]
MEAL_PRESETS = {
    "Douglass et al. 2024: gap ≥ 300 s, any size": (300.0, 1),
    "FED3 paper (2021): gap < 60 s, ≥ 5 pellets": (60.0, 5),
    "Short meals: gap < 60 s, ≥ 2 pellets": (60.0, 2),
    "Custom": None,
}
SMOOTH = {"Off": 0, "3 bins": 3, "5 bins": 5, "30-min running mean (Douglass et al. 2024)": -30}

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
    for s in (60, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400):
        if span_s / s <= target:
            return s
    return 86400 * max(1, int(span_s / 86400 / target))


def _elapsed(v):
    sgn = "−" if v < 0 else ""
    v = abs(v)
    return f"{sgn}{int(v // 3600):02d}:{int(v % 3600 // 60):02d}"


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
    name = "after first pellet" if "first pellet" in alignment else "into recording"

    def f(x):
        return _elapsed(x), f"{_elapsed(x)}:{int(abs(x) % 60):02d} {name}"
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
        short = lab(v)[0]
        if alignment.startswith("Time of day"):
            if v % 86400 == 0 or v == vals[0]:
                short = f"Day {int(v // 86400) + 1}<br>{short}"
            if lights_on is not None:
                z = ((v / 3600.0) - lights_on) % 24
                short += f"<br><span style='color:#64748b'>ZT {z:g}</span>"
        out.append(short)
    return vals, out


def axis_title(alignment):
    return {"Clock": "Clock time", "Time of day": "Time of day (recordings from different days overlaid)",
            "Time since recording": "Time since each recording started (HH:MM)",
            "Time since first pellet": "Time since each animal's first pellet (HH:MM)"}[
        next(k for k in ("Clock", "Time of day", "Time since recording", "Time since first pellet")
             if alignment.startswith(k))]


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


def combined_figure(S, F, stat, alignment, ref, lights_on, darks, show_each, sig_label, fed_label, height):
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
            name = (f"{g} " if len(groups) > 1 else "") + ("signal" if M_ is S else "food") + f" (n = {len(sel)})"
            _band_traces(fig, M_["x"], c, lo, hi, n, col, name, row=row, show=(M_ is S or len(groups) > 1))
        r += 1
    ref_S = S or F
    for row in range(1, rows + 1):
        for a, b in darks:
            fig.add_vrect(x0=a, x1=b, fillcolor=_DARK, line_width=0, layer="below", row=row, col=1)
    _apply_axis(fig, ref_S, alignment, ref, lights_on, rows=list(range(1, rows + 1)))
    fig.update_xaxes(title_text=axis_title(alignment), row=rows, col=1)
    if S:
        fig.update_yaxes(title_text=sig_label, row=1, col=1)
        fig.add_hline(y=0, line=dict(color="#94a3b8", width=0.8, dash="dot"), row=1, col=1)
    if F:
        fig.update_yaxes(title_text=fed_label, row=rows, col=1, rangemode="tozero")
    return _style(fig, height)


def each_figure(S, F_counts, members, alignment, ref, lights_on, darks, layout, sig_label, show_bars, height_mult):
    names = S["rows"]
    k = len(names)
    if layout == "One graph":
        fig = go.Figure()
        for i, nm in enumerate(names):
            col = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(x=S["x"], y=S["M"][i], mode="lines", name=S["animals"][i],
                                     line=dict(color=col, width=1.4),
                                     hovertemplate=f"{S['animals'][i]}: %{{y:.3g}}<extra></extra>"))
        for a, b in darks:
            fig.add_vrect(x0=a, x1=b, fillcolor=_DARK, line_width=0, layer="below")
        _apply_axis(fig, S, alignment, ref, lights_on)
        fig.update_xaxes(title_text=axis_title(alignment))
        fig.update_yaxes(title_text=sig_label)
        return _style(fig, int(560 * height_mult))
    fig = make_subplots(rows=k, cols=1, shared_xaxes=True, vertical_spacing=min(0.035, 0.25 / max(k, 1)),
                        specs=[[{"secondary_y": True}] for _ in range(k)],
                        subplot_titles=[f"{S['animals'][i]} · {S['groups'][i]}" if len(set(S["groups"])) > 1
                                        else f"{S['animals'][i]}  ·  {S['rows'][i]}" for i in range(k)])
    by_name = {m["name"]: m for m in members}
    for i in range(k):
        row = i + 1
        y = S["M"][i]
        fig.add_trace(go.Scatter(x=S["x"], y=y, mode="lines", name=S["animals"][i], showlegend=False,
                                 line=dict(color=_SIG, width=1.2),
                                 hovertemplate=f"{S['animals'][i]}: %{{y:.3g}}<extra></extra>"),
                      row=row, col=1, secondary_y=False)
        if show_bars and F_counts is not None:
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
    _apply_axis(fig, S, alignment, ref, lights_on, rows=list(range(1, k + 1)))
    fig.update_xaxes(title_text=axis_title(alignment), row=k, col=1)
    fig.update_yaxes(title_text=sig_label, row=(k + 1) // 2, col=1, secondary_y=False)
    fig.update_annotations(font=dict(size=12, color="#334155"))
    return _style(fig, int((150 * k + 140) * height_mult))


def heatmap_figure(S, F, alignment, ref, lights_on, sig_label, centred, height_mult):
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
    for lim, b in ((3 * 3600, 1.0), (8 * 3600, 2.0), (30 * 3600, 10.0), (72 * 3600, 30.0)):
        if span_s <= lim:
            return b
    return 60.0


def render(results, fed, roi, prepare_df, max_points=12_000, height_mult=1.0, config=None):
    if len(results) < 1:
        st.info("Load two or more recordings to analyse them together.")
        return
    ss = st.session_state
    animals = list(fed.animals) if fed is not None and not fed.events.empty else []

    st.markdown('<div class="pp-note">Analyse several recordings together: the group mean with its spread, '
                'every recording on the same time axis, heatmaps, and the meal response with each animal as one '
                'data point. ' + REFERENCES + "</div>", unsafe_allow_html=True)

    # ---- members --------------------------------------------------------
    rows = []
    for i, r in enumerate(results):
        sid = (r.get("settings") or {}).get("subject_ID", "")
        guess = fed3.guess_animal(r["name"], animals, sid) if animals else None
        rows.append({"include": True, "recording": str(r["name"]),
                     "animal": guess or fed3.animal_from_name(r["name"]) or "(none)", "group": "All"})
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

    # ---- settings -------------------------------------------------------
    first = next(r for r in results if str(r["name"]) in set(ed["recording"]))
    kinds = [k for k in fed_view.signal_kinds(first["df"], roi) if k != fed_view.RAW_BOTH]
    c = st.columns([1.5, 1.3, 1.6, 1.2])
    alignment = c[0].selectbox("Line recordings up by", ga.ALIGNMENTS, key="grp_align",
                               help="Clock time: same moment (animals recorded together). Time of day: recordings "
                                    "from different days overlaid by time of day (longitudinal). Time since start "
                                    "or since the first pellet: each animal's own zero.")
    kind = c[1].selectbox("Signal", kinds, index=next((i for i, k in enumerate(kinds) if k.startswith("dF/F")), 0),
                          key="grp_kind")
    norm = c[2].selectbox("Normalise each recording", ga.NORMS, key="grp_norm")
    stat = c[3].selectbox("Group line and band", ga.STATS, key="grp_stat")
    view = st.radio("Group view", VIEWS, horizontal=True, key="grp_view")

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
        base_min = e[3].number_input("Baseline period (min)", 1.0, 600.0, 30.0, 1.0, key="grp_basemin",
                                     help="For the baseline normalisations: the first minutes of each recording, "
                                          "or the minutes before the first pellet.")
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

    # ---- build members --------------------------------------------------
    members, problems = [], []
    by_name = {str(r["name"]): r for r in results}
    for _, row in ed.iterrows():
        r = by_name[row["recording"]]
        df = prepare_df(r["df"])
        if df is None or df.empty:
            problems.append(f"{row['recording']}: no data in the visual window")
            continue
        try:
            t, y, unit, _ = gt._panel_series(df, roi, kind, 0.0)
        except Exception as ex:
            problems.append(f"{row['recording']}: {kind} not available ({ex})")
            continue
        cs = gt.clock_start_of(r["df"], r.get("settings"))
        try:
            ctrl = fed_view._control_405(r["df"], roi)
            ctrl = ctrl if ctrl[0] is not None else None
        except Exception:
            ctrl = None
        an = row["animal"] if row["animal"] not in (None, "", "(none)") else None
        members.append(ga.make_member(row["recording"], an, ss["grp_groups"][row["recording"]], t, y, cs,
                                      fed if animals else None, sessions,
                                      (r.get("settings") or {}).get("fit_R2"), ctrl, mg / 1000.0))
    for p in problems:
        st.markdown(f'<div class="pp-note warn">{p}</div>', unsafe_allow_html=True)
    if not members:
        return
    unit = gt._panel_series(prepare_df(first["df"]), roi, kind, 0.0)[2]
    norm_short = {"z-score (whole recording)": "z", "Robust z (median / MAD)": "robust z",
                  "z-score to a baseline period": "z vs baseline", "Subtract baseline mean": f"Δ {unit}".strip(),
                  "As recorded": unit, "Scale 0–1 (1st–99th percentile)": "0–1 scaled"}[norm]
    sig_label = f"{kind.split(' (')[0]} ({norm_short})" if norm_short else kind
    centred = norm.startswith(("z", "Robust", "Subtract"))

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
    g1 = st.columns([1.2, 1.6, 1.4, 1.3])
    bin_opts = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0]
    dflt_bin = _default_bin(span)
    bin_min = g1[0].select_slider("Bin (min)", bin_opts, value=dflt_bin, key=f"grp_bin_{int(dflt_bin * 10)}")
    smooth = g1[1].selectbox("Smoothing", list(SMOOTH), key="grp_smooth")
    fed_mode = g1[2].selectbox("Food shown as", ga.FED_MODES, key="grp_fedmode") \
        if animals and view == "Combined (group mean)" else "Pellets per bin"
    show_each = g1[3].checkbox("Show each animal faintly", True, key="grp_each") \
        if view in ("Combined (group mean)", "Meal-aligned (group)") else False
    k = SMOOTH[smooth]
    k = max(1, int(round(30.0 / bin_min))) if k < 0 else max(1, k)

    notes = [ga.METHOD_NOTES.get(norm, ""), ga.METHOD_NOTES.get(stat, "")]
    if view == "Combined (group mean)" and animals:
        notes.append(ga.METHOD_NOTES.get(fed_mode, ""))
    st.caption(" ".join(n for n in notes if n))

    S = ga.signal_matrix(members, alignment, norm, bin_min * 60.0, k, base_min, ref, per_animal)
    if S is None:
        st.warning("None of the selected recordings can be lined up this way (e.g. no pellets for "
                   "'Time since first pellet').")
        return
    if S["skipped"]:
        st.markdown(f'<div class="pp-note warn">Left out (cannot be aligned this way): '
                    f'{", ".join(S["skipped"])}</div>', unsafe_allow_html=True)
    lab = axis_labeler(alignment, ref, lights_on)
    x0, x1 = S["e0"], S["e0"] + S["n_bins"] * S["bin_s"]
    darks = dark_bands(alignment, x0, x1, ref, lights_on, lights_off) if shade else []
    h = height_mult
    clock_for_ruler = ref if alignment.startswith("Clock") else None

    if view == "Combined (group mean)":
        F = ga.fed_matrix(fed_members, alignment, bin_min * 60.0, fed_mode, ref, per_animal=per_animal) \
            if animals else None
        fed_label = {"Pellets per bin": f"pellets / {bin_min:g} min", "Cumulative intake (g)": "cumulative g",
                     "Cumulative, % of own total": "% of own total", "Animals eating (%)": "% of animals eating",
                     "Eating rate (pellets/h, rolling)": "pellets / h"}[fed_mode]
        fig = combined_figure(S, F, stat, alignment, ref, lights_on if alignment.startswith("Time of day") else None,
                              darks, show_each, sig_label, fed_label, int(760 * h))
        _x_hover(fig, lab)
        gt.chart_block(fig, key="grp_combined", max_points=max_points, clock_start=clock_for_ruler, config=config)
        _downloads(S, F, stat, lab, "combined")

    elif view == "Each recording":
        layout = st.radio("Layout", ["Stacked panels", "One graph"], horizontal=True, key="grp_layout")
        bars = st.checkbox("Pellets per bin behind each trace", True, key="grp_bars") \
            if layout == "Stacked panels" and animals else False
        Fc = ga.fed_matrix(members, alignment, bin_min * 60.0, "Pellets per bin", ref, per_animal=per_animal) \
            if bars else None
        fig = each_figure(S, Fc, members, alignment, ref, lights_on if alignment.startswith("Time of day") else None,
                          darks, layout, sig_label, bars, h)
        _x_hover(fig, lab)
        gt.chart_block(fig, key="grp_each_fig", max_points=max_points * 2, clock_start=clock_for_ruler,
                       config=config)
        st.caption("Ticks above each trace mark single pellets; shaded areas are the dark phase.")

    elif view == "Heatmap":
        F = ga.fed_matrix(members, alignment, bin_min * 60.0, "Pellets per bin", ref, per_animal=per_animal) \
            if animals else None
        fig = heatmap_figure(S, F, alignment, ref, lights_on if alignment.startswith("Time of day") else None,
                             sig_label, centred, h)
        st.plotly_chart(fig, use_container_width=True, config=config, key="grp_heat")

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
        st.caption(f"Meals: pellets less than {gap:g} s apart, at least {minp} pellet(s). Satiety ratio: minutes "
                   "without eating after a meal per pellet in that meal. Hours logged excludes times the FED "
                   "was not recording. % pellets in dark uses the lights-off/on hours under More settings.")
        st.download_button("Download feeding pattern (CSV)", tab.to_csv(index=False).encode(),
                           "feeding_pattern_per_animal.csv", "text/csv")


def _downloads(S, F, stat, lab, tag):
    out = pd.DataFrame({"x_seconds": S["x"], "time": [lab(v)[1] for v in S["x"]]})
    for i, a in enumerate(S["animals"]):
        out[f"signal {a} [{S['groups'][i]}]"] = S["M"][i]
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
    st.caption(f"Grey: baseline {base[0]:g} to {base[1]:g} s; blue: response {resp[0]:g} to {resp[1]:g} s. Each "
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
