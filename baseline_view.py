#!/usr/bin/env python3
"""
baseline_view.py
================

Streamlit pieces for "Normalise to a baseline period" in the one-recording
view (the group view has its own controls; both use baseline.py).

    cfg = controls(key, t_min, t_max, clock_start)      the switch, length, end time
    df2 = apply(df, roi, cfg, stats_df)                 z and dF/F re-referenced
    shade(fig, cfg)                                     baseline band on a figure
    response_panel(df, roi, cfg, key, clock_start)      peak vs baseline, with a graph

The baseline END is set by clicking any graph of the recording ("📍 Set
baseline end") or by typing a time. Its value lives in
st.session_state[pick_key(key)] in seconds of recording time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import baseline as bl
import graph_tools as gt
import theme as th

BAND = "rgba(37, 99, 235, 0.10)"
EDGE = "#2563eb"


def pick_key(key):
    return f"{key}_bl_end"


def _default_length(duration):
    if duration < 3600:
        return 0, 5
    if duration < 6 * 3600:
        return 0, 30
    return 1, 0


def _parse_time(text, clock_start, t_min, t_max):
    """Recording time ('1:02:03', '62:03', '3723') or clock time ('3:05 PM', '15:05')."""
    import group_view as gv
    txt = str(text or "").strip()
    if not txt:
        return None
    looks_clock = any(s in txt.lower() for s in ("am", "pm")) or txt.count("-") >= 2
    if clock_start is not None and looks_clock:
        return gv.parse_clock_time(txt, clock_start)
    v = gv.parse_recording_time(txt)
    if v is not None:
        return v
    if clock_start is not None:
        return gv.parse_clock_time(txt, clock_start)
    return None


def _fmt(sec, clock_start):
    s = gt.fmt_hms(sec, 1)
    if clock_start is not None:
        s += f" ({gt.fmt_clock(clock_start, sec, date=False, decimals=0)})"
    return s


def controls(key, t_min, t_max, clock_start=None):
    """The baseline settings for one recording. Returns a dict (on=False when off)."""
    duration = max(t_max - t_min, 1.0)
    pk = pick_key(key)
    with st.container(border=True):
        on = st.toggle("Normalise to a baseline period", key=f"{key}_bl_on",
                       help="Use the mean and SD of a period you choose (e.g. the minutes before food was "
                            "given) instead of the whole recording. The z-score graphs then show z against "
                            "that baseline, and dF/F shows the % change from the baseline level.")
        h0, m0 = _default_length(duration)
        txt_key = f"{key}_bl_txt"
        clicked = st.session_state.get(pk)
        if clicked is not None and st.session_state.get(f"{key}_bl_synced") != clicked:
            st.session_state[txt_key] = gt.fmt_hms(clicked, 1)       # a click on a graph fills the field
            st.session_state[f"{key}_bl_synced"] = clicked
        length, end, dff_too = 0.0, None, True
        if on:
            top = st.columns([0.8, 0.8, 2.0, 1.8])
            hours = top[0].number_input("Length: hours", 0, 96, h0, 1, key=f"{key}_bl_h")
            mins = top[1].number_input("minutes", 0.0, 59.99, float(m0), 0.5, key=f"{key}_bl_m")
            length = hours * 3600.0 + mins * 60.0
            typed = top[2].text_input("Baseline ends at", key=txt_key,
                                      placeholder="click a graph, or e.g. 1:30:00 or 3:05 PM",
                                      help="Recording time (H:MM:SS) or clock time with AM/PM (3:05 PM). Easiest: "
                                           "switch on '📍 Set baseline end' above any graph of this recording and "
                                           "click the moment the baseline should end, e.g. just before the event.")
            end = _parse_time(typed, clock_start, t_min, t_max) if typed else None
            if typed and end is None:
                top[2].caption(":red[Time not understood]")
            if end is not None:
                st.session_state[pk] = end
                st.session_state[f"{key}_bl_synced"] = end
            dff_too = top[3].checkbox("dF/F as % change from the baseline", True, key=f"{key}_bl_dff",
                                      help="Also re-reference dF/F: 0 % = the baseline level. Off: dF/F stays as "
                                           "computed; only the z-score changes.")
        cfg = {"on": bool(on and end is not None and length > 0), "end": end, "length": length,
               "dff_too": bool(dff_too), "key": key, "wanted": bool(on)}
        if on and end is None:
            st.caption("Now click the graph below where the baseline should end (**📍 Set baseline end** is on), "
                       "or type the time.")
        elif on and length <= 0:
            st.caption("Give the baseline a length.")
        elif cfg["on"]:
            start = end - length
            msg = f"Baseline: **{_fmt(max(start, t_min), clock_start)} → {_fmt(end, clock_start)}**"
            if start < t_min - 1:
                msg += (f". :orange[It starts {gt.fmt_hms(t_min - start, 0)} before the recording does, so only "
                        f"{gt.fmt_hms(max(end - t_min, 0), 0)} of it has data.]")
            st.caption(msg)
    return cfg


def stats_for(df, roi, cfg):
    """Baseline mean/SD of dF/F (%) for this recording, or None."""
    col = f"{roi}_dFF_percent"
    if col not in df.columns:
        if f"{roi}_dFF" not in df.columns:
            return None
        y = 100.0 * pd.to_numeric(df[f"{roi}_dFF"], errors="coerce").to_numpy(float)
    else:
        y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    t = pd.to_numeric(df["time_sec"], errors="coerce").to_numpy(float)
    return bl.baseline_stats(t, y, cfg["end"], cfg["length"])


def apply(df, roi, cfg, stats):
    """Copy of df with {roi}_z_dFF = z against the baseline and, if chosen, dF/F as % change from it."""
    if not cfg.get("on") or stats is None or not stats.get("ok"):
        return df
    out = df.copy()
    col = f"{roi}_dFF_percent"
    if col in out.columns:
        y = pd.to_numeric(out[col], errors="coerce").to_numpy(float)
    else:
        y = 100.0 * pd.to_numeric(out[f"{roi}_dFF"], errors="coerce").to_numpy(float)
    out[f"{roi}_z_dFF"] = bl.z_to_baseline(y, stats)
    if cfg.get("dff_too"):
        pct = bl.pct_change_dff(y, stats["mean"])
        out[col] = pct
        if f"{roi}_dFF" in out.columns:
            out[f"{roi}_dFF"] = pct / 100.0
    return out


def shade(fig, cfg):
    """Blue band over the baseline and a line at its end, on every panel of the figure."""
    if not cfg.get("on"):
        return fig
    a, b = cfg["end"] - cfg["length"], cfg["end"]
    try:
        fig.add_vrect(x0=a, x1=b, fillcolor=BAND, line_width=0, layer="below", row="all", col="all")
        fig.add_vline(x=b, line=dict(color=EDGE, width=1.4, dash="dot"), row="all", col="all")
    except Exception:
        fig.add_vrect(x0=a, x1=b, fillcolor=BAND, line_width=0, layer="below")
        fig.add_vline(x=b, line=dict(color=EDGE, width=1.4, dash="dot"))
    fig.add_annotation(x=b, y=1.0, yref="paper", text="baseline end", showarrow=False, xanchor="right",
                       yanchor="bottom", font=dict(size=10, color=EDGE))
    return fig


def response_panel(df_raw, roi, cfg, key, clock_start=None, height=430):
    """'Response after the baseline': peak, z against the baseline only vs the whole window, return time."""
    t = pd.to_numeric(df_raw["time_sec"], errors="coerce").to_numpy(float)
    col = f"{roi}_dFF_percent"
    y = (pd.to_numeric(df_raw[col], errors="coerce").to_numpy(float) if col in df_raw.columns else
         100.0 * pd.to_numeric(df_raw[f"{roi}_dFF"], errors="coerce").to_numpy(float))
    end, length = cfg["end"], cfg["length"]
    left = float(np.nanmax(t) - end)
    if left <= 1:
        st.info("The baseline ends at the end of the recording; there is no response to measure.")
        return None
    c = st.columns([1, 1, 1, 2])
    resp_min = c[0].number_input("Response window (min)", 0.05, max(0.1, left / 60.0),
                                 float(min(max(length / 60.0, 1.0), left / 60.0)), 0.5, key=f"{key}_bl_resp",
                                 help="How long after the baseline end to look for the peak and its return.")
    direction = c[1].selectbox("Look for", ["Larger change", "Peak (up)", "Dip (down)"], key=f"{key}_bl_dir")
    dt = float(np.nanmedian(np.diff(t))) if t.size > 1 else 1.0
    smooth = c[2].number_input("Smooth (s)", 0.0, 600.0, float(max(1.0, 2 * dt)), 0.5, key=f"{key}_bl_sm",
                               help="Running mean before finding the peak, so one noisy sample is not "
                                    "taken as the peak.")
    res = bl.peak_vs_baseline(t, y, end, length, resp_min * 60.0,
                              {"Larger change": "auto", "Peak (up)": "up", "Dip (down)": "down"}[direction],
                              smooth)
    if not res.get("ok"):
        st.warning(res.get("message"))
        return res
    m = st.columns(5)
    m[0].metric("Peak change", f"{res['peak_change_pct']:+.2f} %",
                help="dF/F at the peak as % change from the baseline level.")
    m[1].metric("Peak z, baseline", f"{res['peak_z']:+.2f}",
                help="(peak − baseline mean) ÷ baseline SD.")
    m[2].metric("z, whole window", f"{res['peak_z_whole_window']:+.2f}",
                help="What you get if the mean and SD are taken over the whole window shown "
                     "(baseline and response together), as a whole-recording z-score does.")
    m[3].metric("Time to peak", gt.fmt_hms(res["peak_time_s"], 1))
    m[4].metric("Back to baseline", gt.fmt_hms(res["return_s"], 1) if res.get("return_s") is not None else "not yet",
                help="First time after the peak that the smoothed signal is within 1 SD of the baseline mean.")
    st.caption(bl.describe(res, length) + (" " + res["message"] if res.get("message") else ""))

    a, b = end - length, end + resp_min * 60.0
    sel = (t >= a) & (t <= b)
    ys = bl.smooth_by_time(t, y, smooth)[sel]
    tt = t[sel] - end
    zb = (ys - res["baseline_mean_pct"]) / res["baseline_sd_pct"]
    zw = (ys - res["whole_mean_pct"]) / res["whole_sd_pct"]
    fig = go.Figure()
    fig.add_vrect(x0=-length, x1=0, fillcolor=BAND, line_width=0, layer="below")
    fig.add_hrect(y0=-1, y1=1, fillcolor="rgba(100,116,139,0.10)", line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=tt, y=zw, mode="lines", name="z over baseline + response",
                             line=dict(color="#94a3b8", width=1.6, dash="dash")))
    fig.add_trace(go.Scatter(x=tt, y=zb, mode="lines", name="z against the baseline only",
                             line=dict(color="#0072B2", width=2.2),
                             customdata=np.c_[bl.pct_change_dff(ys, res["baseline_mean_pct"])],
                             hovertemplate="%{x:.1f} s · z %{y:.2f} · %{customdata[0]:+.2f} % from baseline"
                                           "<extra></extra>"))
    fig.add_trace(go.Scatter(x=[res["peak_time_s"]], y=[res["peak_z"]], mode="markers+text", name="peak",
                             marker=dict(size=11, color="#dc2626", symbol="diamond"),
                             text=[f"z {res['peak_z']:+.1f}"], textposition="top center", showlegend=False))
    if res.get("return_s") is not None:
        fig.add_vline(x=res["return_s"], line=dict(color="#16a34a", width=1.2, dash="dot"),
                      annotation_text="back within 1 SD", annotation_position="top")
    fig.add_vline(x=0, line=dict(color=EDGE, width=1.4, dash="dot"))
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=60, r=20, t=40, b=50),
                      legend=dict(orientation="h", y=1.08, x=0), hovermode="x unified",
                      xaxis_title="seconds from the end of the baseline", yaxis_title="z")
    st.plotly_chart(fig, width="stretch", key=f"{key}_bl_fig")
    th.explain("Why the two z-scores differ",
               "**Blue (solid):** z against the baseline only, the band shaded blue. 0 is the baseline mean, and "
               "±1 (grey band) its normal ups and downs.\n\n**Grey (dashed):** the same signal z-scored over "
               "baseline + response together, the way a whole-recording z-score works. The response itself then "
               "pulls the mean towards it and inflates the SD, so the same peak looks smaller. Report the "
               "baseline-only number when the question is how far the signal moved from where it was before.")
    row = {k: res.get(k) for k in ("direction", "peak_time_s", "peak_dff_pct", "peak_change_pct", "peak_z",
                                   "peak_z_whole_window", "mean_z", "auc_z", "onset_s", "return_s",
                                   "duration_s", "baseline_mean_pct", "baseline_sd_pct", "baseline_n")}
    row.update({"baseline_start_s": a, "baseline_end_s": end, "response_window_s": resp_min * 60.0,
                "smooth_s": smooth})
    st.download_button("Download these numbers (CSV)", pd.DataFrame([row]).to_csv(index=False).encode(),
                       file_name=f"{key}_peak_vs_baseline.csv", mime="text/csv", key=f"{key}_bl_dl")
    return res


__all__ = ["controls", "stats_for", "apply", "shade", "response_panel", "pick_key"]
