#!/usr/bin/env python3
"""
fed_view.py
===========

"Feeding (FED3)" view of the app: photometry and FED3 food intake on the
shared PC clock.

  1. Photometry above, food below   signal on top; pellets per bin and
                                    cumulative intake underneath, same time axis
  2. One graph                      signal with every pellet marked ON the trace
                                    and the eating rate on a second axis
  3. Aligned to eating              signal around each pellet or meal onset:
                                    mean +/- SEM, every event, and a test of
                                    whether the change is bigger than around
                                    random times
  4. All animals                    cumulative intake of every animal together,
                                    plus a summary table

All graphs carry the time ruler (which also counts pellets between its two
points) and the peaks/troughs tool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import fed3
import graph_tools as gt

_PELLET = "#b45309"
_MEAL = "rgba(180,83,9,0.13)"
_GAP = "rgba(148,163,184,0.22)"
_CUM = "#7c3aed"
_SIG = "#0072B2"


def _signal(df, roi, kind, smooth_min):
    t, y, unit, _ = gt._panel_series(df, roi, kind, smooth_min * 60.0)
    return t, y, unit


def _gaps(cov_sec, t0, t1):
    """Complement of the FED coverage intervals inside [t0, t1]."""
    out, cur = [], t0
    for a, b in sorted(cov_sec):
        if a > cur:
            out.append((cur, min(a, t1)))
        cur = max(cur, b)
    if cur < t1:
        out.append((cur, t1))
    return [(a, b) for a, b in out if b - a > 1]


def _add_gaps(fig, gaps, rows=None):
    for a, b in gaps:
        kw = dict(x0=a, x1=b, fillcolor=_GAP, line_width=0, layer="below")
        if rows:
            for r in rows:
                fig.add_vrect(row=r, col=1, **kw)
        else:
            fig.add_vrect(**kw)
    if gaps:
        a, b = max(gaps, key=lambda g: g[1] - g[0])
        fig.add_annotation(x=(a + b) / 2, y=0.99, yref="paper", text="FED not logging",
                           showarrow=False, font=dict(size=11, color="#475569"), yanchor="top")


def _band(x0, x1, lo, hi):
    """Many rectangles as one NaN-separated filled trace (fast to draw)."""
    x0 = np.asarray(x0, float)
    x1 = np.asarray(x1, float)
    bx = np.column_stack([x0, x0, x1, x1, np.full_like(x0, np.nan)]).ravel()
    by = np.tile([lo, hi, hi, lo, np.nan], x0.size)
    return bx, by


def _clock_axis(fig, t0, t1, cs, rows=None):
    vals, labels = gt.time_ticks(t0, t1, cs)
    kw = dict(tickvals=vals, ticktext=labels, showgrid=True, gridcolor="#e5e7eb",
              range=[t0, t1])
    if rows:
        for r in rows:
            fig.update_xaxes(row=r, col=1, **kw)
    else:
        fig.update_xaxes(**kw)


def _style(fig, height):
    fig.update_layout(template="plotly_white", height=height, hovermode="x unified",
                      dragmode="pan", margin=dict(l=64, r=64, t=50, b=50),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                                  font=dict(color="#111827", size=12)),
                      font=dict(color="#111827"))
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7")
    return fig


# =============================================================================
# FIGURES
# =============================================================================

def stacked_figure(t, y, sig_label, pellets_s, meals, bins, gaps, t0, t1, cs,
                   grams_each, bin_min, height):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.58, 0.42],
                        specs=[[{}], [{"secondary_y": True}]],
                        subplot_titles=(sig_label, f"Food: pellets per {bin_min:g} min and cumulative"))
    if t is not None:
        fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=sig_label,
                                 line=dict(color=_SIG, width=1.1)), row=1, col=1)
        top = float(np.nanmax(y)) if np.isfinite(y).any() else 1.0
        bot = float(np.nanmin(y)) if np.isfinite(y).any() else 0.0
        if len(meals):
            bx, by = _band(meals["onset"], meals["offset"] + 5, bot, top)
            fig.add_trace(go.Scatter(x=bx, y=by, fill="toself", fillcolor=_MEAL, line_width=0,
                                     mode="lines", hoverinfo="skip", name="meals",
                                     showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=pellets_s, y=np.full(pellets_s.size, top + 0.04 * (top - bot)),
                                 mode="markers", name="pellet taken",
                                 marker=dict(symbol="line-ns-open", size=10, color=_PELLET,
                                             line=dict(width=1.4, color=_PELLET)),
                                 hovertemplate="pellet<extra></extra>"), row=1, col=1)
    ok = bins["pellets"].notna()
    fig.add_trace(go.Bar(x=bins.loc[ok, "mid"], y=bins.loc[ok, "pellets"], width=bin_min * 60 * 0.9,
                         name=f"pellets / {bin_min:g} min", marker_color=_PELLET, opacity=0.75,
                         customdata=np.c_[bins.loc[ok, "pellets"] * grams_each],
                         hovertemplate="%{y:.0f} pellets (%{customdata[0]:.2f} g)<extra></extra>"),
                  row=2, col=1)
    cum = np.arange(1, pellets_s.size + 1)
    fig.add_trace(go.Scatter(x=np.r_[t0, pellets_s], y=np.r_[0, cum] * grams_each, mode="lines",
                             line=dict(color=_CUM, width=2, shape="hv"), name="cumulative food (g)",
                             hovertemplate="%{y:.2f} g eaten so far<extra></extra>"),
                  row=2, col=1, secondary_y=True)
    _add_gaps(fig, gaps, rows=[1, 2])
    _clock_axis(fig, t0, t1, cs, rows=[1, 2])
    fig.update_xaxes(title_text="Clock time" if cs is not None else "Elapsed", row=2, col=1)
    fig.update_yaxes(title_text=sig_label, row=1, col=1)
    fig.update_yaxes(title_text="pellets / bin", row=2, col=1, rangemode="tozero")
    fig.update_yaxes(title_text="cumulative (g)", row=2, col=1, secondary_y=True,
                     rangemode="tozero", showgrid=False)
    return _style(fig, height)


def overlay_figure(t, y, sig_label, pellets_s, retrieval_s, meals, rate_grid, rate,
                   gaps, t0, t1, cs, rate_win_min, height):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=sig_label,
                             line=dict(color=_SIG, width=1.1)), secondary_y=False)
    good = np.isfinite(y)
    at = np.interp(pellets_s, t[good], y[good]) if good.any() and pellets_s.size else np.array([])
    hover = [f"pellet {i + 1}" + (f" · taken {r:.1f} s after drop" if np.isfinite(r) else "")
             for i, r in enumerate(retrieval_s)]
    fig.add_trace(go.Scatter(x=pellets_s, y=at, mode="markers", name="pellet taken",
                             marker=dict(size=7, color=_PELLET, line=dict(color="#fff", width=0.8)),
                             text=hover, hovertemplate="%{text}<br>signal %{y:.3g}<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=rate_grid, y=rate, mode="lines", name=f"eating rate (pellets/h, {rate_win_min:g}-min window)",
                             line=dict(color=_CUM, width=1.6, dash="dot")), secondary_y=True)
    _add_gaps(fig, gaps)
    _clock_axis(fig, t0, t1, cs)
    fig.update_xaxes(title_text="Clock time" if cs is not None else "Elapsed")
    fig.update_yaxes(title_text=sig_label, secondary_y=False)
    fig.update_yaxes(title_text="pellets per hour", secondary_y=True, rangemode="tozero", showgrid=False,
                     tickmode="auto", nticks=6, tickformat=".0f")
    return _style(fig, height)


def aligned_figure(mat, tax, unit_label, base, resp, title, height):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.5, 0.5],
                        subplot_titles=(f"Mean ± SEM (n = {mat.shape[0]})", "Every event"))
    m = np.nanmean(mat, axis=0)
    n = np.sum(np.isfinite(mat), axis=0)
    sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if mat.shape[0] > 1 else np.zeros_like(m)
    fig.add_trace(go.Scatter(x=np.r_[tax, tax[::-1]], y=np.r_[m + sem, (m - sem)[::-1]],
                             fill="toself", fillcolor="rgba(0,114,178,0.18)", line_width=0,
                             hoverinfo="skip", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=tax, y=m, mode="lines", name="mean", line=dict(color=_SIG, width=2)),
                  row=1, col=1)
    for (a, b), col, lab in ((base, "rgba(100,116,139,0.15)", "baseline"),
                             (resp, "rgba(180,83,9,0.15)", "response")):
        fig.add_vrect(x0=a, x1=b, fillcolor=col, line_width=0, row=1, col=1,
                      annotation_text=lab, annotation_position="top left",
                      annotation_font_size=10)
    fig.add_trace(go.Heatmap(z=mat, x=tax, y=np.arange(1, mat.shape[0] + 1), colorscale="RdBu_r",
                             zmid=0, colorbar=dict(title=unit_label, len=0.45, y=0.22)), row=2, col=1)
    for r in (1, 2):
        fig.add_vline(x=0, line=dict(color="#111827", width=1.3, dash="dash"), row=r, col=1)
    fig.update_xaxes(title_text="seconds from event", row=2, col=1)
    fig.update_yaxes(title_text=unit_label, row=1, col=1)
    fig.update_yaxes(title_text="event #", row=2, col=1, autorange="reversed")
    _style(fig, height)
    fig.update_layout(title=dict(text=title, y=0.99, yanchor="top"), margin=dict(t=110),
                      legend=dict(y=1.06))
    return fig


# =============================================================================
# MAIN VIEW
# =============================================================================

def render(results, fed, roi, prepare_df, max_points=12_000, height_mult=1.0, config=None):
    if fed is None or fed.events.empty:
        st.info("Upload FED3 data in the sidebar (**FED3 feeding data**): the zipped "
                "FED3_data folder, session folders' files, or individual *_data.csv files.")
        return

    for w in fed.warnings:
        st.markdown(f'<div class="pp-note warn">{w}</div>', unsafe_allow_html=True)
    with st.expander("What was read", expanded=False):
        for n in fed.notes:
            st.write("- " + n)
        if not fed.sessions.empty:
            st.dataframe(fed.sessions, hide_index=True, use_container_width=True)
        cnt = fed.events.groupby(["animal", "session", "kind"]).size().unstack(fill_value=0)
        st.dataframe(cnt, use_container_width=True)

    # ---- settings -----------------------------------------------------------
    labels = fed.sessions["session"].tolist() if not fed.sessions.empty else sorted(fed.events["session"].unique())
    default_sessions = [s for s in labels if not str(s).lower().startswith("test")]
    c = st.columns([2.2, 1, 1, 1])
    sessions = c[0].multiselect("FED sessions to use", labels, default=default_sessions, key="fed_sessions",
                                help="Sessions whose name starts with 'test' are left out by default.")
    grams_each = c[1].number_input("Pellet mass (mg)", 1.0, 1000.0, fed3.PELLET_MG_DEFAULT, 1.0,
                                   key="fed_mg") / 1000.0
    bin_min = c[2].selectbox("Intake bins (min)", [5, 10, 15, 30, 60], index=1, key="fed_bin")
    rate_win = c[3].selectbox("Eating-rate window (min)", [10, 30, 60, 120], index=1, key="fed_ratewin")

    # ---- which recording is which animal ------------------------------------
    animals = fed.animals
    rec_names = [str(r["name"]) for r in results]
    st.markdown("**Recording ↔ animal**")
    mcols = st.columns(max(1, min(4, len(results) or 1)))
    rec_of = {}
    for i, r in enumerate(results):
        guess = fed3.animal_from_name(r["name"]) or fed3.animal_from_name(
            (r.get("settings") or {}).get("subject_ID", ""))
        opts = ["(none)"] + animals
        a = mcols[i % len(mcols)].selectbox(rec_names[i], opts,
                                            index=opts.index(guess) if guess in opts else 0,
                                            key=f"fed_map_{i}")
        if a != "(none)":
            if a in rec_of:
                st.warning(f"Two recordings are matched to {a}; the second is used.")
            rec_of[a] = i
    if not results:
        st.caption("No photometry loaded; showing food intake only.")

    view = st.radio("Feeding view", ["Photometry above, food below", "One graph: signal + eating",
                                     "Aligned to eating", "All animals"],
                    horizontal=True, key="fed_view")

    ref_clock = None
    if rec_of:
        df0 = prepare_df(results[next(iter(rec_of.values()))]["df"])
        ref_clock = gt.clock_start_of(df0, results[next(iter(rec_of.values()))].get("settings"))
    if ref_clock is None:
        ref_clock = fed.events["time"].min().floor("h")

    if view == "All animals":
        _all_animals(fed, sessions, grams_each, bin_min, ref_clock, max_points, height_mult, config, results, rec_of, roi, prepare_df)
        return

    animal = st.selectbox("Animal", animals, key="fed_animal")
    pel = fed.times(animal, "pellet", sessions)
    pel_rows = fed.events[(fed.events["animal"] == animal) & (fed.events["kind"] == "pellet")
                          & fed.events["session"].isin(sessions)].sort_values("time")
    cov = fed.coverage_of(animal, sessions)

    if animal in rec_of:
        res = results[rec_of[animal]]
        df = prepare_df(res["df"])
        cs = gt.clock_start_of(df, res.get("settings"))
        kinds = [k for k in gt.panel_types(df, roi) if k.startswith(("z-score", "dF/F"))] or list(gt.panel_types(df, roi))
        sc = st.columns([1.4, 1, 3])
        kind = sc[0].selectbox("Signal", kinds, key="fed_kind")
        smooth = sc[1].number_input("Running-mean window (min)", 0.5, 240.0, 30.0, 1.0, key="fed_smooth") \
            if "running" in kind else 30.0
        t, y, unit = _signal(df, roi, kind, smooth)
        if cs is None:
            st.error("This recording has no clock time, so it cannot be lined up with the FED "
                     "data. Re-run the analysis with this version of the app.")
            return
        t0, t1 = float(np.nanmin(t)), float(np.nanmax(t))
        sig_label = kind + (f" ({unit})" if unit and unit not in kind else "")
    else:
        df = None
        t = y = None
        cs = ref_clock
        st.caption(f"No photometry recording is matched to {animal}; showing intake only.")
        sig_label = ""
        ps = fed3.to_seconds(pel, cs)
        t0 = float(min(ps.min(), *fed3.to_seconds(cov["start"], cs))) if len(ps) else 0.0
        t1 = float(max(ps.max(), *fed3.to_seconds(cov["stop"], cs))) if len(ps) else 3600.0

    pellets_s = fed3.to_seconds(pel, cs)
    inside = (pellets_s >= t0) & (pellets_s <= t1)
    n_out = int((~inside).sum())
    pellets_s = pellets_s[inside]
    retrieval = pel_rows["retrieval_s"].to_numpy(float)[inside] if "retrieval_s" in pel_rows else np.full(pellets_s.size, np.nan)
    cov_s = list(zip(fed3.to_seconds(cov["start"], cs), fed3.to_seconds(cov["stop"], cs)))
    gaps = _gaps(cov_s, t0, t1)
    mc = st.columns([1, 1, 4])
    gap_s = mc[0].number_input("Meal: max gap between pellets (s)", 5.0, 3600.0, 60.0, 5.0, key="fed_mealgap",
                               help="Pellets closer together than this are one meal.")
    min_p = mc[1].number_input("Meal: min pellets", 1, 50, 2, 1, key="fed_mealmin")
    meals = fed3.find_meals(pellets_s, gap_s, int(min_p))

    # Summary line
    logged_h = sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in cov_s) / 3600.0
    st.markdown(
        f"**{animal}**: {pellets_s.size} pellets = **{pellets_s.size * grams_each:.2f} g** "
        f"in {logged_h:.1f} h of FED logging ({pellets_s.size / logged_h if logged_h else 0:.1f} pellets/h) · "
        f"{len(meals)} meals, median {np.median(meals['n_pellets']) if len(meals) else 0:.0f} pellets"
        + (f" · {n_out} pellet(s) fall outside the photometry recording and are not shown" if n_out else ""))

    h = int(height_mult * 640)
    key = f"fed_{animal}_{view[:6]}"
    counts = [("pellets", pellets_s, grams_each)]

    if view == "Photometry above, food below":
        bins = fed3.intake_bins(pellets_s, t0, t1, bin_min * 60.0, cov_s)
        fig = stacked_figure(t, y, sig_label, pellets_s, meals, bins, gaps, t0, t1, cs,
                             grams_each, bin_min, int(height_mult * 760))
        gt.chart_block(fig, key=key, max_points=max_points, clock_start=cs, config=config,
                       peaks=t is not None, event_counts=counts)
        st.caption("Grey bands: the FED was not being logged, so intake there is unknown (not zero). "
                   "Ticks above the trace and shaded bands mark pellets and meals. Hover the bars for "
                   "pellets and grams per bin; use the 📏 ruler to count pellets between any two times.")
    elif view == "One graph: signal + eating":
        if t is None:
            st.info("Match a photometry recording to this animal to see the signal.")
            return
        grid = np.linspace(t0, t1, 3000)
        rate = fed3.rolling_rate(pellets_s, grid, rate_win * 60.0)
        cov_mask = np.zeros(grid.size, bool)
        for a, b in cov_s:
            cov_mask |= (grid >= a) & (grid <= b)
        rate = np.where(cov_mask, rate, np.nan)
        fig = overlay_figure(t, y, sig_label, pellets_s, retrieval, meals, grid, rate, gaps, t0, t1, cs,
                             rate_win, h)
        gt.chart_block(fig, key=key, max_points=max_points, clock_start=cs, config=config,
                       peaks=True, event_counts=counts)
        st.caption("Each orange dot is a pellet taken, placed on the signal at that moment. Dotted line: "
                   "eating rate. Turn on ▲▼ Peaks to see whether the highest peaks and lowest troughs "
                   "sit near eating; the 📏 ruler gives the time from a pellet to a peak.")
    else:
        if t is None:
            st.info("Match a photometry recording to this animal to align its signal to eating.")
            return
        _aligned(t, y, sig_label, pellets_s, meals, cov_s, pel_rows[inside].reset_index(drop=True),
                 cs, animal, grams_each, h, config, df=df, roi=roi)


def _control_405(df, roi):
    """
    The 405 channel as a z-score after removing its slow drift (the same
    0.001 Hz baseline idea as F0). If it ALSO changes at eating, movement,
    chewing or blood flow may be behind the 465 response.
    """
    import photometry_core as pc
    col = next((c for c in (f"{roi}_uv_raw", f"{roi}_uv_filt", f"{roi}_uv_filt_10Hz_lowpass")
                if c in df.columns), None)
    if col is None:
        return None, None
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    u = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    ok = np.isfinite(u)
    if ok.sum() < 100:
        return None, None
    fs = 1.0 / float(np.median(np.diff(t)))
    filled = np.interp(np.arange(u.size), np.flatnonzero(ok), u[ok])
    slow = pc.lowpass(filled, fs, 0.001)
    return t, np.where(ok, pc.zscore(filled - slow), np.nan)


def _aligned(t, y, sig_label, pellets_s, meals, cov_s, pel_rows, cs, animal, grams_each, h, config,
             df=None, roi=None):
    a = st.columns([1.4, 1, 1, 1, 1])
    what = a[0].radio("Align to", ["Meal onsets", "Every pellet"], key="fed_align_to",
                      help="Douglass et al. 2025 compared −100…−50 s with 0…50 s for meals and "
                           "−40…−20 s with 0…20 s for single feeding bouts; those are the defaults.")
    meal_mode = what == "Meal onsets"
    pre = a[1].number_input("Seconds before", 10.0, 3600.0, 300.0 if meal_mode else 120.0, 10.0, key=f"fed_pre_{meal_mode}")
    post = a[2].number_input("Seconds after", 10.0, 3600.0, 600.0 if meal_mode else 180.0, 10.0, key=f"fed_post_{meal_mode}")
    b0 = a[3].text_input("Baseline window (s)", "-100,-50" if meal_mode else "-40,-20", key=f"fed_base_{meal_mode}")
    r0 = a[4].text_input("Response window (s)", "0,50" if meal_mode else "0,20", key=f"fed_resp_{meal_mode}")
    try:
        base = tuple(float(v) for v in b0.split(","))[:2]
        resp = tuple(float(v) for v in r0.split(","))[:2]
        assert base[0] < base[1] and resp[0] < resp[1]
    except Exception:
        st.error("Windows must look like  -100,-50  (start,end in seconds).")
        return
    sub = st.checkbox("Subtract each event's baseline", value=True, key="fed_sub")
    events = meals["onset"].to_numpy(float) if meal_mode else pellets_s
    if events.size == 0:
        st.info("No events in this recording window.")
        return
    mat, tax, kept = fed3.aligned_matrix(t, y, events, pre, post, baseline=base if sub else None)
    if mat.shape[0] == 0:
        st.info("No event has a complete window inside the recording.")
        return
    stats = fed3.permutation_test(t, y, kept, base, resp, allowed_sec=cov_s, n_perm=2000)
    unit = ("Δ " if sub else "") + sig_label
    fig = aligned_figure(mat, tax, unit, base, resp,
                         f"{animal}: {sig_label} around {'meal onsets' if meal_mode else 'pellets'}", int(h * 1.05))
    ct, cu = _control_405(df, roi) if df is not None else (None, None)
    cstats = None
    if ct is not None:
        cmat, ctax, _ = fed3.aligned_matrix(ct, cu, kept, pre, post, baseline=base if sub else None)
        cstats = fed3.permutation_test(ct, cu, kept, base, resp, allowed_sec=cov_s, n_perm=2000)
        # Drawn only on a z-score axis: a z-scored 405 next to dF/F in % would
        # mix units. The verdict below is shown either way.
        if cmat.shape[0] and "z" in sig_label.lower():
            fig.add_trace(go.Scatter(x=ctax, y=np.nanmean(cmat, axis=0), mode="lines",
                                     name="405 control (z, drift removed)",
                                     line=dict(color="#64748b", width=1.6, dash="dot")), row=1, col=1)
    gt.chart_block(fig, key=f"fed_al_{animal}_{meal_mode}", max_points=0, clock_start=None,
                   config=config, peaks=False)
    if stats["n"] >= 3 and np.isfinite(stats["p"]):
        direction = "rises" if stats["observed"] > stats["null_mean"] else "falls"
        verdict = ("a larger change than around random times" if stats["p"] < 0.05
                   else "not different from random times")
        st.markdown(
            f"**Response − baseline = {stats['observed']:+.3g}** (mean over {stats['n']} events). "
            f"Around random times with the FED logging: {stats['null_mean']:+.3g} "
            f"(95% range {stats['ci95'][0]:+.3g} to {stats['ci95'][1]:+.3g}). "
            f"The signal **{direction}** after eating, **{verdict}** "
            f"(permutation p = {stats['p']:.3g}, 2,000 random sets).")
        st.caption("Random times are drawn only when the FED was logging and away from real "
                   "events, so the comparison is fair against slow drifts in the signal.")
        if cstats is not None and np.isfinite(cstats["p"]):
            c_eff = cstats["observed"] - cstats["null_mean"]
            msg = (f"<b>405 control:</b> {c_eff:+.3g} z relative to random times (p = {cstats['p']:.3g}). ")
            if cstats["p"] < 0.05 and np.sign(c_eff) == np.sign(stats["observed"] - stats["null_mean"]):
                msg += ("The 405 channel moves the SAME way at eating, so movement, chewing or "
                        "blood flow may explain part of this response.")
                kind_ = "warn"
            elif cstats["p"] < 0.05:
                msg += ("The 405 channel moves the OPPOSITE way. That is what a calcium signal does "
                        "when 405 nm is not exactly isosbestic, but it is also what made this "
                        "recording's 405 fit slope negative; interpret with care.")
                kind_ = "warn"
            else:
                msg += "405 does not change at eating, so the response is unlikely to be a movement artefact."
                kind_ = "ok"
            st.markdown(f'<div class="pp-note {kind_}">{msg}</div>', unsafe_allow_html=True)
    else:
        st.caption("Fewer than 3 events with complete windows; no test run.")

    # per-event table
    d_resp = fed3.window_means(t, y, kept, *resp)
    d_base = fed3.window_means(t, y, kept, *base)
    tbl = pd.DataFrame({
        "#": np.arange(1, kept.size + 1),
        "clock time": [gt.fmt_clock(cs, v)[:19] for v in kept],
        "baseline mean": np.round(d_base, 4), "response mean": np.round(d_resp, 4),
        "change": np.round(d_resp - d_base, 4),
    })
    if meal_mode:
        mm = meals.set_index("onset").reindex(kept)
        tbl.insert(2, "pellets in meal", mm["n_pellets"].to_numpy())
        tbl.insert(3, "meal (g)", np.round(mm["n_pellets"].to_numpy() * grams_each, 3))
        tbl.insert(4, "meal length (s)", np.round(mm["duration_s"].to_numpy(), 1))
    st.dataframe(tbl, hide_index=True, use_container_width=True, height=min(420, 40 + 35 * len(tbl)))
    st.download_button("Download this table (CSV)", data=lambda: tbl.to_csv(index=False).encode(),
                       file_name=f"{animal}_{'meals' if meal_mode else 'pellets'}_aligned.csv",
                       mime="text/csv", key=f"fed_dl_{animal}_{meal_mode}")


def _all_animals(fed, sessions, grams_each, bin_min, ref, max_points, height_mult, config,
                 results, rec_of, roi, prepare_df):
    fig = go.Figure()
    rows = []
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#7c3aed", "#111827"]
    t_lo, t_hi = np.inf, -np.inf
    for i, a in enumerate(fed.animals):
        ps = fed3.to_seconds(fed.times(a, "pellet", sessions), ref)
        cov = fed.coverage_of(a, sessions)
        cov_s = list(zip(fed3.to_seconds(cov["start"], ref), fed3.to_seconds(cov["stop"], ref)))
        if ps.size == 0:
            continue
        t_lo, t_hi = min(t_lo, ps.min(), *(c[0] for c in cov_s)), max(t_hi, ps.max(), *(c[1] for c in cov_s))
        fig.add_trace(go.Scatter(x=np.r_[ps[0], ps], y=np.r_[0, np.arange(1, ps.size + 1)] * grams_each,
                                 mode="lines", line=dict(shape="hv", width=2, color=palette[i % len(palette)]),
                                 name=a, hovertemplate=f"{a}: %{{y:.2f}} g<extra></extra>"))
        meals = fed3.find_meals(ps, 60.0, 2)
        logged = sum(b - c for c, b in cov_s) / 3600.0
        rows.append({"animal": a, "pellets": ps.size, "food (g)": round(ps.size * grams_each, 3),
                     "hours logged": round(logged, 2),
                     "pellets / h logged": round(ps.size / logged, 2) if logged else np.nan,
                     "meals (≥2 pellets, gap ≤60 s)": len(meals),
                     "median meal (pellets)": float(np.median(meals["n_pellets"])) if len(meals) else np.nan,
                     "photometry": results[rec_of[a]]["name"] if a in rec_of else "—"})
    if not rows:
        st.info("No pellets in the selected sessions.")
        return
    allcov = []
    for a in fed.animals:
        cov = fed.coverage_of(a, sessions)
        allcov += list(zip(fed3.to_seconds(cov["start"], ref), fed3.to_seconds(cov["stop"], ref)))
    _add_gaps(fig, _gaps(allcov, t_lo, t_hi))
    _clock_axis(fig, t_lo, t_hi, ref)
    fig.update_yaxes(title_text="cumulative food (g)", rangemode="tozero")
    fig.update_xaxes(title_text="Clock time")
    _style(fig, int(height_mult * 520))
    gt.chart_block(fig, key="fed_all", max_points=max_points, clock_start=ref, config=config)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Grey: no FED logging for any animal. Meals in this table use the default "
               "definition; change it per animal in the other views.")
