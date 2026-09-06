import streamlit as st
from pathlib import Path
import tempfile
import subprocess
import sys
import zipfile
import io
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared, validated maths + UI modules.
import photometry_core as pc
import lickometer as lk
import theme as th

# =============================================================================
# PAGE STYLE: SMALL HEADER + WIDE GRAPH WORKSPACE
# =============================================================================

st.set_page_config(
    page_title="Photometry Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A single scoped stylesheet lives in theme.py, and base colours come from
# .streamlit/config.toml. The three overlapping !important blocks that used to
# sit here were what broke the sidebar multiselect chips.
th.inject_theme()

th.page_header(
    "Fiber Photometry Analysis",
    "pyPhotometry .ppd / .csv  \u00b7  motion correction, dF/F, z-score, lickometer",
    mark="FP",
)

# =============================================================================
# JONES-STYLE PROCESSING SETTINGS FOR CSV INPUTS
# =============================================================================

# Defaults now come from photometry_core so the .ppd route and the .csv route
# cannot drift apart again. Previously this file used a 1 Hz low-pass, a 60 s
# rolling regression and an F0 taken from the 405 channel, while ana.py used a
# 10 Hz low-pass, a global fit and an F0 taken from the 465 channel. The same
# animal therefore gave two different answers depending on which file you
# uploaded.
LOWPASS_HZ = pc.DEFAULTS["lowpass_hz"]          # 10.0
FILTER_ORDER = pc.DEFAULTS["filter_order"]
REGRESSION_WINDOW_SEC = 60.0                     # retained for legacy settings display
BASELINE_PERCENTILE = pc.DEFAULTS["f0_percentile"]
EXPORT_HZ_DEFAULT = 1.0

PLOTLY_CONFIG = {
    "scrollZoom": False,  # prevents mouse/trackpad scroll from zooming graphs
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

DEFAULT_GRAPH_COLORS = dict(th.TRACE_COLORS)


def graph_color(name):
    """Return user-selected graph color, or a sensible default."""
    return st.session_state.get(f"color_{name}", DEFAULT_GRAPH_COLORS.get(name, "#111111"))


# =============================================================================
# HELPERS
# =============================================================================

def seconds_to_hms(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"



def parse_hms_to_seconds(value, default_seconds=0.0, label="time"):
    """
    Parse HH:MM:SS, MM:SS, or SS into seconds.

    Examples:
      06:26:51 -> 23211 seconds
      26:51    -> 1611 seconds
      90       -> 90 seconds

    The app uses seconds internally for plotting, but all user-facing time inputs
    are HH:MM:SS-style strings.
    """
    if value is None:
        return float(default_seconds), None
    raw = str(value).strip()
    if raw == "":
        return float(default_seconds), None

    try:
        parts = raw.split(":")
        if len(parts) == 3:
            h, m, s = parts
            total = int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            total = int(m) * 60 + float(s)
        elif len(parts) == 1:
            total = float(parts[0])
        else:
            raise ValueError
        if total < 0:
            raise ValueError
        return float(total), None
    except Exception:
        return float(default_seconds), f"Invalid {label}: '{raw}'. Use HH:MM:SS, MM:SS, or seconds."


def seconds_to_hours_for_json(seconds):
    return float(seconds) / 3600.0


def time_label_from_seconds(seconds):
    return seconds_to_hms(seconds)


def make_time_ticks(min_sec, max_sec, n=None):
    """
    Return clean hour-based x-axis ticks.

    The graph x-axis uses seconds internally so hover can still show HH:MM:SS,
    but the visible tick labels are simple elapsed hours: 0, 1, 2, 3, ...
    """
    interval = 3600.0  # 1 hour
    min_sec = float(min_sec) if np.isfinite(min_sec) else 0.0
    max_sec = float(max_sec) if np.isfinite(max_sec) else max(min_sec + interval, interval)
    if max_sec <= min_sec:
        max_sec = min_sec + interval

    start_tick = max(0.0, np.floor(min_sec / interval) * interval)
    end_tick = np.ceil(max_sec / interval) * interval
    ticks = np.arange(start_tick, end_tick + interval * 0.5, interval, dtype=float)

    if ticks.size < 2 and (max_sec - min_sec) <= interval:
        ticks = np.arange(0.0, max(interval, max_sec) + 1800.0, 1800.0, dtype=float)
        labels = [f"{t / 3600:g}" for t in ticks]
    else:
        labels = [str(int(round(t / 3600.0))) for t in ticks]

    return ticks.tolist(), labels


def apply_hms_xaxis(fig, df, rows=None):
    """Show clean elapsed-hour x-axis marks while keeping HH:MM:SS hover."""
    if "time_sec" not in df.columns or df.empty:
        return fig
    min_sec = float(np.nanmin(df["time_sec"]))
    max_sec = float(np.nanmax(df["time_sec"]))
    tickvals, ticktext = make_time_ticks(min_sec, max_sec)

    if rows is None:
        fig.update_xaxes(
            title_text="Time (hours)",
            tickmode="array",
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
            automargin=True,
        )
    else:
        bottom_row = max(rows)
        for row in rows:
            show_labels = row == bottom_row
            fig.update_xaxes(
                title_text="Time (hours)" if show_labels else "",
                tickmode="array",
                tickvals=tickvals,
                ticktext=ticktext,
                showticklabels=show_labels,
                tickangle=0,
                automargin=True,
                row=row,
                col=1,
            )
    return fig


def get_start_end_seconds(item):
    """Read visual annotation intervals, supporting old *_hr keys and new *_sec keys."""
    if "start_sec" in item:
        start = float(item.get("start_sec", 0.0))
    else:
        start = float(item.get("start_hr", 0.0)) * 3600.0

    if "end_sec" in item and item.get("end_sec") is not None:
        end = float(item.get("end_sec"))
    elif item.get("end_hr") is not None:
        end = float(item.get("end_hr")) * 3600.0
    else:
        end = None
    return start, end


def safe_name(name):
    return (
        str(name).replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def find_time_column(df):
    candidates = ["time_sec", "elapsed_seconds", "time_seconds", "time_s", "Time(s)", "Time", "time"]
    for c in candidates:
        if c in df.columns:
            return c
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        raise ValueError("Could not find a numeric time column in this CSV.")
    return numeric[0]


def standardize_time(df):
    df = df.copy()
    time_col = find_time_column(df)
    if time_col != "time_sec":
        df["time_sec"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")

    df = df.dropna(subset=["time_sec"]).sort_values("time_sec").reset_index(drop=True)

    # Only convert milliseconds to seconds when the input truly looks like ms.
    #
    # IMPORTANT:
    # A 48-hour recording in seconds has max(time_sec) around 172800.
    # The old rule divided any long recording by 1000, which made 48 hours
    # display as ~2 minutes 53 seconds. Do not use max(time_sec) alone.
    time_col_lower = str(time_col).lower()
    time_values = df["time_sec"].to_numpy(dtype=float)
    diffs = np.diff(time_values)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    median_step = float(np.nanmedian(diffs)) if diffs.size else np.nan

    looks_like_ms_name = any(token in time_col_lower for token in ["time_ms", "ms", "msec", "millisecond"])
    looks_like_ms_values = (
        np.isfinite(median_step)
        and median_step > 1.0
        and float(np.nanmax(time_values)) > 1_000_000
    )

    if looks_like_ms_name or looks_like_ms_values:
        df["time_sec"] = df["time_sec"] / 1000.0

    df["elapsed_hours"] = df["time_sec"] / 3600.0
    df["elapsed_hhmmss"] = [seconds_to_hms(x) for x in df["time_sec"]]
    return df


def lowpass(values, fs):
    nyquist = fs / 2.0
    if LOWPASS_HZ >= nyquist:
        return values.copy()
    sos = butter(FILTER_ORDER, LOWPASS_HZ, btype="lowpass", fs=fs, output="sos")
    return sosfiltfilt(sos, values)


def estimate_fs(time_sec):
    diffs = np.diff(time_sec)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return EXPORT_HZ_DEFAULT
    return float(1.0 / np.median(diffs))


def rolling_filtered_to_raw_fit(uv_raw, sig_raw, uv_filt, sig_filt, fs):
    window = max(int(round(REGRESSION_WINDOW_SEC * fs)), 5)
    minp = max(3, window // 5)

    x = pd.Series(uv_filt)
    y = pd.Series(sig_filt)
    xy = pd.Series(uv_filt * sig_filt)
    x2 = pd.Series(uv_filt * uv_filt)

    mx = x.rolling(window, center=True, min_periods=minp).mean()
    my = y.rolling(window, center=True, min_periods=minp).mean()
    mxy = xy.rolling(window, center=True, min_periods=minp).mean()
    mx2 = x2.rolling(window, center=True, min_periods=minp).mean()

    cov = mxy - mx * my
    var = mx2 - mx * mx

    slope = (cov / var.replace(0, np.nan)).bfill().ffill().to_numpy()
    intercept = (my - slope * mx).bfill().ffill().to_numpy()

    if np.isnan(slope).any() or np.isnan(intercept).any():
        good = np.isfinite(uv_filt) & np.isfinite(sig_filt)
        global_slope, global_intercept = np.polyfit(uv_filt[good], sig_filt[good], 1)
        slope = np.where(np.isfinite(slope), slope, global_slope)
        intercept = np.where(np.isfinite(intercept), intercept, global_intercept)

    uv_fit = slope * uv_raw + intercept
    delta_f = sig_raw - uv_fit
    return uv_fit, delta_f


def find_raw_channel_columns(df, roi):
    columns = list(df.columns)
    pairs = [
        (f"{roi}_uv_raw", f"{roi}_sig_raw"),
        (f"{roi}_iso", f"{roi}_sig"),
        (f"{roi}_405", f"{roi}_465"),
        ("BLA_uv_raw", "BLA_sig_raw"),
        ("BLA_iso", "BLA_sig"),
        ("405_dark_subtracted_volts", "465_dark_subtracted_volts"),
        ("isosbestic_405_dark_subtracted_volts", "signal_465_dark_subtracted_volts"),
        ("405", "465"),
    ]
    for uv_col, sig_col in pairs:
        if uv_col in columns and sig_col in columns:
            return uv_col, sig_col

    uv_like = [c for c in columns if ("405" in c.lower() or "iso" in c.lower() or "uv" in c.lower())]
    sig_like = [c for c in columns if ("465" in c.lower() or "sig" in c.lower() or "gcamp" in c.lower())]
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
    uv_like = [c for c in uv_like if c in numeric_cols]
    sig_like = [c for c in sig_like if c in numeric_cols]
    if uv_like and sig_like:
        return uv_like[0], sig_like[0]
    return None, None


def processed_columns_exist(df, roi):
    candidates = [f"{roi}_deltaF", f"{roi}_dFF", f"{roi}_z_dFF", f"{roi}_dff", f"{roi}_z_dff"]
    return any(c in df.columns for c in candidates)


def normalize_processed_column_names(df, roi):
    df = df.copy()
    if f"{roi}_dFF" not in df.columns and f"{roi}_dff" in df.columns:
        df[f"{roi}_dFF"] = df[f"{roi}_dff"]
    if f"{roi}_z_dFF" not in df.columns and f"{roi}_z_dff" in df.columns:
        df[f"{roi}_z_dFF"] = df[f"{roi}_z_dff"]
    if f"{roi}_dFF" in df.columns and f"{roi}_z_dFF" not in df.columns:
        y = pd.to_numeric(df[f"{roi}_dFF"], errors="coerce")
        sd = y.std(ddof=1)
        df[f"{roi}_z_dFF"] = (y - y.mean()) / sd if sd and np.isfinite(sd) else np.nan
    return df


def process_raw_csv_dataframe(df, roi, settings=None):
    """
    Process a raw two-channel CSV through the shared, validated pipeline.

    Three bugs in the previous version of this function are fixed here.

    1. F0 came from the isosbestic channel:
           f0 = percentile(uv_raw, 10);  dFF = 100 * deltaF / f0
       F0 is baseline fluorescence of the *signal*. Taking it from the raw 405
       channel divides by a different physical quantity on a different scale, so
       every dF/F magnitude produced by the CSV route was wrong. dF/F is now
       deltaF divided by a baseline derived from the 465 channel.

    2. The regression was fitted on filtered traces but applied to raw ones:
           uv_fit = slope * uv_raw + intercept;  delta_f = sig_raw - uv_fit
       That reinjected all the high-frequency noise the filter had just removed.
       deltaF is now computed entirely from the filtered traces.

    3. The isosbestic was fitted with ordinary least squares. Real calcium
       transients are genuine 465-only divergences, and OLS treats them as error
       and pulls the fit toward them, subtracting away part of the real signal.
       The default is now IRLS (Huber), per Keevers & Jean-Richard-dit-Bressel
       (2025), with OLS still selectable.
    """
    df = standardize_time(df)
    uv_col, sig_col = find_raw_channel_columns(df, roi)
    if uv_col is None or sig_col is None:
        raise ValueError(
            "Could not identify raw 405/isosbestic and 465/signal columns. "
            "Use columns like time_sec,BLA_iso,BLA_sig or time_sec,BLA_uv_raw,BLA_sig_raw."
        )

    uv_raw = pd.to_numeric(df[uv_col], errors="coerce").to_numpy(dtype=float)
    sig_raw = pd.to_numeric(df[sig_col], errors="coerce").to_numpy(dtype=float)

    good = np.asarray(np.isfinite(df["time_sec"]) & np.isfinite(uv_raw) & np.isfinite(sig_raw))
    df = df.loc[good].reset_index(drop=True)
    uv_raw, sig_raw = uv_raw[good], sig_raw[good]

    fs = estimate_fs(df["time_sec"].to_numpy())
    res = pc.process_photometry(sig_raw, uv_raw, fs, settings=settings)

    out = pd.DataFrame({
        "time_sec": df["time_sec"].to_numpy(),
        "elapsed_hours": df["time_sec"].to_numpy() / 3600.0,
        f"{roi}_uv_raw": uv_raw,
        f"{roi}_sig_raw": sig_raw,
        f"{roi}_uv_filt": res["ctl_filt"],
        f"{roi}_sig_filt": res["sig_filt"],
        f"{roi}_uv_fit": res["ctl_fit"],
        f"{roi}_deltaF": res["deltaF"],
        f"{roi}_F0_trace": res["F0_trace"],
        f"{roi}_dFF": res["dFF"],
        f"{roi}_z_dFF": res["z_dFF"],
    })
    out["elapsed_hhmmss"] = [seconds_to_hms(x) for x in out["time_sec"]]

    cfg = res["settings"]
    settings_out = {
        "input_type": "csv_raw",
        "uv_column_used": uv_col,
        "sig_column_used": sig_col,
        "sampling_rate_estimated_hz": fs,
        "median_filter_sec": cfg["median_filter_sec"],
        "lowpass_hz": cfg["lowpass_hz"],
        "filter_order": cfg["filter_order"],
        "fit_method": cfg["fit_method"],
        "fit_slope": res["slope"],
        "fit_intercept": res["intercept"],
        "f0_method": cfg["f0_method"],
        "F0_median": res["F0"],
        "dff_units": cfg["dff_units"],
        "zscore_mode": cfg["zscore_mode"],
        "deltaF_formula": "deltaF = filtered_465 - fit(filtered_405)",
        "dFF_formula": "dFF = deltaF / F0(465-derived)",
    }
    return out, settings_out


def load_processed_csv_dataframe(df, roi):
    df = standardize_time(df)
    df = normalize_processed_column_names(df, roi)
    if "elapsed_hours" not in df.columns:
        df["elapsed_hours"] = df["time_sec"] / 3600.0
    return df, {"input_type": "csv_processed", "note": "CSV already contained processed columns; no reprocessing was done."}


def filter_visual_window(df, start_sec, end_sec):
    if start_sec is None or end_sec is None:
        return df
    return df[(df["time_sec"] >= start_sec) & (df["time_sec"] <= end_sec)].copy()


def apply_crop_to_results(results, roi, start_sec, end_sec, recompute_stats=True):
    """
    Physically crop every loaded recording and re-zero its time axis.

    This is different from the "visual crop window", which only changes what is
    drawn. Cropping here changes the data: samples outside the window are
    dropped and the requested start becomes t = 0. Crop a 48 h recording at
    24:00:00 and what was hour 24 is now hour 0, hour 25 is hour 1, and the
    downloaded CSV matches what is on screen.

    recompute_stats
        After cropping, the retained window IS the recording, so its z-score
        should describe that window. With this on, z-dF/F is recomputed from the
        cropped dF/F. With it off, the original whole-session z-score is kept,
        which is what you want if you are comparing a cropped segment against
        statistics defined over the full session.
    """
    out = []
    for r in results:
        df, offset = pc.crop_and_rezero(r["df"], start_sec, end_sec, rezero=True)
        if df.empty:
            continue
        if recompute_stats and f"{roi}_dFF" in df.columns:
            dff = pd.to_numeric(df[f"{roi}_dFF"], errors="coerce").to_numpy(float)
            df[f"{roi}_z_dFF"] = pc.zscore(dff)
        if "elapsed_hhmmss" in df.columns:
            df["elapsed_hhmmss"] = [seconds_to_hms(x) for x in df["time_sec"]]
        new = dict(r)
        new["df"] = df
        if r.get("digital_events"):
            lo = float(start_sec) if start_sec is not None else -np.inf
            hi = float(end_sec) if end_sec is not None else np.inf
            new["digital_events"] = {
                ch: t[(t >= lo) & (t <= hi)] - offset
                for ch, t in r["digital_events"].items()
            }
        new["settings"] = dict(r.get("settings", {}))
        new["settings"].update({
            "cropped": True,
            "crop_start_sec_original_timeline": start_sec,
            "crop_end_sec_original_timeline": end_sec,
            "time_offset_subtracted_sec": offset,
            "zscore_recomputed_on_crop": bool(recompute_stats),
        })
        out.append(new)
    return out


def shift_events_for_crop(events, offset):
    """Move event annotations onto the re-zeroed timeline."""
    if not offset:
        return events
    shifted = []
    for e in events or []:
        e2 = dict(e)
        for key in ("start_sec", "end_sec"):
            if e2.get(key) is not None:
                e2[key] = float(e2[key]) - float(offset)
        shifted.append(e2)
    return shifted


def finite_limits(values, pad_fraction=0.06):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 1.0
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_fraction
    return lo - pad, hi + pad


def get_auto_limits(results, roi):
    def collect(col):
        vals = []
        for r in results:
            df = r["df"]
            if col in df.columns:
                vals.append(df[col].to_numpy())
        return np.concatenate(vals) if vals else np.array([])

    return {
        "raw": finite_limits(np.concatenate([collect(f"{roi}_uv_raw"), collect(f"{roi}_sig_raw")])),
        "fit": finite_limits(np.concatenate([collect(f"{roi}_sig_raw"), collect(f"{roi}_uv_fit")])),
        "deltaF": finite_limits(collect(f"{roi}_deltaF")),
        "dFF": finite_limits(collect(f"{roi}_dFF")),
        "z_dFF": finite_limits(collect(f"{roi}_z_dFF")),
    }


def parse_limit_text(text_value, default):
    if text_value is None or str(text_value).strip() == "":
        return default
    try:
        parts = [float(x.strip()) for x in str(text_value).split(",")]
        if len(parts) != 2 or parts[0] >= parts[1]:
            return default
        return (parts[0], parts[1])
    except Exception:
        return default


def apply_y_range(fig, row, limit):
    if limit is not None:
        fig.update_yaxes(range=[limit[0], limit[1]], row=row, col=1)


def signal_columns(df, roi):
    """Columns that should be hidden when visual-only time periods are removed."""
    candidates = [
        f"{roi}_uv_raw",
        f"{roi}_sig_raw",
        f"{roi}_uv_fit",
        f"{roi}_deltaF",
        f"{roi}_dFF",
        f"{roi}_z_dFF",
        f"{roi}_dff",
        f"{roi}_z_dff",
    ]
    return [c for c in candidates if c in df.columns]


def apply_visual_exclusions(df, roi, exclusions, mode="blank"):
    """
    Hide selected time periods visually only.
    The full processed CSV stored in session_state is not changed.
    """
    if not exclusions:
        return df

    df2 = df.copy()
    mask_all = np.zeros(len(df2), dtype=bool)
    for ex in exclusions:
        start, end = get_start_end_seconds(ex)
        if end is None or end <= start:
            continue
        mask_all |= (df2["time_sec"].to_numpy() >= start) & (df2["time_sec"].to_numpy() <= end)

    if not mask_all.any():
        return df2

    if mode == "remove_rows":
        return df2.loc[~mask_all].copy()

    # Default: blank signal values with NaN, preserving the true time axis.
    for c in signal_columns(df2, roi):
        df2.loc[mask_all, c] = np.nan
    return df2


def hex_to_rgba(hex_color, alpha):
    """Convert #RRGGBB to rgba(r,g,b,a)."""
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) != 7:
        hex_color = "#808080"
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{float(alpha):.3f})"


def add_visual_exclusion_shapes(fig, exclusions, rows, show=True):
    """Light gray marks for visually hidden/cut regions."""
    if not show:
        return fig
    for ex in exclusions:
        start, end = get_start_end_seconds(ex)
        if end is None or end <= start:
            continue
        for row in rows:
            fig.add_vrect(
                x0=start, x1=end,
                fillcolor="rgba(180,180,180,0.13)",
                line_width=0,
                row=row, col=1,
            )
            fig.add_vline(
                x=start, line_width=1, line_dash="dash",
                line_color="rgba(180,180,180,0.40)",
                row=row, col=1,
            )
            fig.add_vline(
                x=end, line_width=1, line_dash="dash",
                line_color="rgba(180,180,180,0.40)",
                row=row, col=1,
            )
    return fig


def add_events_to_fig(fig, events, rows, show_labels=True):
    """
    Add event lines and optional duration shading.
    Events are visual annotations only; they do not change the data.
    """
    if not events:
        return fig

    for ev in events:
        name = str(ev.get("name", "Event")).strip() or "Event"
        start, end = get_start_end_seconds(ev)
        color = ev.get("color", "#ff4b4b")
        alpha = float(ev.get("alpha", 0.65))
        shade_alpha = float(ev.get("shade_alpha", max(0.05, alpha * 0.35)))

        if start is None or not np.isfinite(start):
            continue
        start = float(start)
        has_duration = end is not None and np.isfinite(end) and float(end) > start
        line_color = hex_to_rgba(color, alpha)

        for row in rows:
            if has_duration:
                fig.add_vrect(
                    x0=start, x1=float(end), fillcolor=color, opacity=shade_alpha,
                    line_width=0, row=row, col=1,
                )
                fig.add_vline(
                    x=float(end), line_width=1.4, line_dash="dot",
                    line_color=line_color, row=row, col=1,
                )
            fig.add_vline(
                x=start, line_width=1.4, line_dash="dot",
                line_color=line_color, row=row, col=1,
            )

        if show_labels:
            label = f"<b>{name}</b><br>{seconds_to_hms(start)}"
            if has_duration:
                label += f"–{seconds_to_hms(float(end))}"
            fig.add_annotation(
                x=start, y=1.04, xref="x", yref="paper",
                text=label, showarrow=False,
                font=dict(size=10, color=color), align="center",
                bgcolor="rgba(255,255,255,0.60)",
                bordercolor=color, borderwidth=1, borderpad=2,
            )
    return fig


# NOTE: finish_fig / make_overview_figure / make_raw_independent_figure were
# previously defined twice. The shadowed first copies have been removed; the
# live definitions are in the FIGURES section below.

def make_processed_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=760):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=("Delta-F", "dFF (%) - main normalized trace", "z-scored dFF"))
    x = df["time_sec"]
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F", line=dict(color=graph_color("deltaF"))), row=1, col=1)
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)", line=dict(color=graph_color("dff"))), row=2, col=1)
    if f"{roi}_z_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_z_dFF"], mode="lines", name="z-dFF", line=dict(color=graph_color("raw_z"))), row=3, col=1)
    fig.update_yaxes(title_text="Delta-F", row=1, col=1)
    fig.update_yaxes(title_text="dFF (%)", row=2, col=1)
    fig.update_yaxes(title_text="z-dFF", row=3, col=1)
    apply_hms_xaxis(fig, df, rows=[1, 2, 3])
    if limits:
        apply_y_range(fig, 1, limits.get("deltaF"))
        apply_y_range(fig, 2, limits.get("dFF"))
        apply_y_range(fig, 3, limits.get("z_dFF"))
    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1, 2, 3], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1, 2, 3], show_labels=show_event_labels)
    return finish_fig(fig, height=height)


def make_dual_delta_dff_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=560):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df["time_sec"]
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F", line=dict(color=graph_color("deltaF"))), secondary_y=False)
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)", line=dict(color=graph_color("dff"))), secondary_y=True)
    apply_hms_xaxis(fig, df)
    fig.update_yaxes(title_text="Delta-F", secondary_y=False)
    fig.update_yaxes(title_text="dFF (%)", secondary_y=True)
    if limits:
        if limits.get("deltaF") is not None:
            fig.update_yaxes(range=list(limits["deltaF"]), secondary_y=False)
        if limits.get("dFF") is not None:
            fig.update_yaxes(range=list(limits["dFF"]), secondary_y=True)
    # Single-panel dual-axis graph: add visual markers without row/col args.
    if show_exclusions:
        for ex in exclusions or []:
            start, end = get_start_end_seconds(ex)
            if end is not None and end > start:
                fig.add_vrect(x0=start, x1=end, fillcolor="rgba(180,180,180,0.13)", line_width=0)
    for ev in events or []:
        start, end = get_start_end_seconds(ev)
        if start is None:
            continue
        color = ev.get("color", "#ff4b4b")
        alpha = float(ev.get("alpha", 0.65))
        shade_alpha = float(ev.get("shade_alpha", max(0.05, alpha * 0.35)))
        line_color = hex_to_rgba(color, alpha)
        fig.add_vline(x=float(start), line_width=1.4, line_dash="dot", line_color=line_color)
        if end is not None and np.isfinite(end) and float(end) > float(start):
            fig.add_vrect(x0=float(start), x1=float(end), fillcolor=color, opacity=shade_alpha, line_width=0)
            fig.add_vline(x=float(end), line_width=1.4, line_dash="dot", line_color=line_color)
        if show_event_labels:
            fig.add_annotation(
                x=float(start), y=1.04, xref="x", yref="paper",
                text=f"<b>{ev.get('name','Event')}</b><br>{seconds_to_hms(float(start))}", showarrow=False,
                font=dict(size=10, color=color), bgcolor="rgba(255,255,255,0.60)",
                bordercolor=color, borderwidth=1, borderpad=2,
            )
    return finish_fig(fig, height=height)


def make_zip_from_results(results, roi):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for result in results:
            name = safe_name(result["name"])
            df = result["df"]
            settings = result.get("settings", {})
            z.writestr(f"{name}/processed_output.csv", df.to_csv(index=False))
            z.writestr(f"{name}/analysis_settings.json", json.dumps(settings, indent=2))
            z.writestr(
                f"{name}/README.txt",
                "Visual crop/zoom settings from the webpage are not applied to processed_output.csv.\n"
                "The CSV remains the full processed data for the upload.\n\n"
                f"Main dFF column: {roi}_dFF\n"
                f"Delta-F column: {roi}_deltaF\n"
                f"z-dFF column: {roi}_z_dFF\n",
            )
    buffer.seek(0)
    return buffer.read()



# =============================================================================
# OVERRIDES: light-mode Plotly styling, HH:MM:SS hover, and one main dFF graph
# =============================================================================

def apply_hms_hover(fig, df):
    """Make hover show Time as HH:MM:SS instead of raw seconds."""
    if df is None or "elapsed_hhmmss" not in df.columns:
        return fig
    hms = df["elapsed_hhmmss"].astype(str).to_numpy()
    n = len(hms)
    for trace in fig.data:
        try:
            if hasattr(trace, "x") and trace.x is not None and len(trace.x) == n:
                trace.customdata = hms
                label = trace.name if trace.name else "value"
                trace.hovertemplate = f"Time: %{{customdata}}<br>{label}: %{{y:.6g}}<extra></extra>"
        except Exception:
            pass
    return fig


def finish_fig(fig, height, show_legend=True, df=None):
    if df is not None:
        apply_hms_hover(fig, df)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#111827", size=13),
        height=height,
        margin=dict(l=70, r=35, t=115, b=70),
        hovermode="x unified",
        dragmode="pan",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="left",
            x=0,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#d1d5db",
            borderwidth=1,
        ),
        showlegend=show_legend,
    )
    fig.update_annotations(font=dict(size=14, color="#111827"), yshift=8)
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        linecolor="#9ca3af",
        tickfont=dict(color="#111827", size=11),
        title_font=dict(color="#111827", size=12),
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e5e7eb",
        zeroline=False,
        linecolor="#9ca3af",
        tickfont=dict(color="#111827", size=11),
        title_font=dict(color="#111827", size=12),
        automargin=True,
    )
    return fig


def make_overview_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=850):
    """Overview without any dFF duplicate. The single dFF graph lives in the Main dFF tab."""
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.095,
        subplot_titles=(
            "Raw 465 and 405 together",
            "Correction check: 465 signal and fitted 405",
            "Delta-F corrected signal",
        ),
    )
    x = df["time_sec"]
    if f"{roi}_sig_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465", line=dict(color=graph_color("sig_raw"))), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465", showlegend=False, line=dict(color=graph_color("sig_raw"))), row=2, col=1)
    if f"{roi}_uv_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="UV Raw / 405", line=dict(color=graph_color("uv_raw"))), row=1, col=1)
    if f"{roi}_uv_fit" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_fit"], mode="lines", name="UV Fit / fitted 405", line=dict(color=graph_color("uv_fit"))), row=2, col=1)
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F", line=dict(color=graph_color("deltaF"))), row=3, col=1)

    fig.update_yaxes(title_text="Raw", row=1, col=1)
    fig.update_yaxes(title_text="Fit", row=2, col=1)
    fig.update_yaxes(title_text="Delta-F", row=3, col=1)
    apply_hms_xaxis(fig, df, rows=[1, 2, 3])

    if limits:
        apply_y_range(fig, 1, limits.get("raw"))
        apply_y_range(fig, 2, limits.get("fit"))
        apply_y_range(fig, 3, limits.get("deltaF"))

    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1, 2, 3], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1, 2, 3], show_labels=show_event_labels)
    return finish_fig(fig, height=height, df=df)


def make_raw_independent_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=650):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=("465 nm calcium-dependent signal", "405 nm isosbestic/control signal"),
    )
    x = df["time_sec"]
    if f"{roi}_sig_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="465 raw", line=dict(color=graph_color("sig_raw"))), row=1, col=1)
    if f"{roi}_uv_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="405 raw", line=dict(color=graph_color("uv_raw"))), row=2, col=1)
    fig.update_yaxes(title_text="465 raw", row=1, col=1)
    fig.update_yaxes(title_text="405 raw", row=2, col=1)
    apply_hms_xaxis(fig, df, rows=[1, 2])
    if limits:
        apply_y_range(fig, 1, limits.get("raw"))
        apply_y_range(fig, 2, limits.get("raw"))
    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1, 2], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1, 2], show_labels=show_event_labels)
    return finish_fig(fig, height=height, df=df)


def make_main_dff_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=620):
    """The only dFF graph in the app."""
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Main dFF (%)",))
    x = df["time_sec"]
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)", line=dict(color=graph_color("dff"))), row=1, col=1)
    fig.update_yaxes(title_text="dFF (%)", row=1, col=1)
    apply_hms_xaxis(fig, df, rows=[1])
    if limits:
        apply_y_range(fig, 1, limits.get("dFF"))
    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1], show_labels=show_event_labels)
    return finish_fig(fig, height=height, df=df)


def get_zscore_values(df, roi, source="z-dFF column"):
    """
    Return a z-score trace for the long-term z-score visualization.

    Preferred/default source:
      {roi}_z_dFF if it exists.

    Fallback:
      z-score the dFF or Delta-F column:
          z = (trace - mean(trace)) / SD(trace)

    This is a visualization layer. It does not overwrite the processed CSV.
    """
    if source == "z-dFF column" and f"{roi}_z_dFF" in df.columns:
        z = pd.to_numeric(df[f"{roi}_z_dFF"], errors="coerce").to_numpy(dtype=float)
        label = "z-dFF"
    elif source == "Re-z-score dFF" and f"{roi}_dFF" in df.columns:
        y = pd.to_numeric(df[f"{roi}_dFF"], errors="coerce").to_numpy(dtype=float)
        sd = np.nanstd(y, ddof=1)
        z = (y - np.nanmean(y)) / sd if sd and np.isfinite(sd) else np.full_like(y, np.nan)
        label = "z-score of dFF"
    elif source == "Re-z-score Delta-F" and f"{roi}_deltaF" in df.columns:
        y = pd.to_numeric(df[f"{roi}_deltaF"], errors="coerce").to_numpy(dtype=float)
        sd = np.nanstd(y, ddof=1)
        z = (y - np.nanmean(y)) / sd if sd and np.isfinite(sd) else np.full_like(y, np.nan)
        label = "z-score of Delta-F"
    elif f"{roi}_z_dFF" in df.columns:
        z = pd.to_numeric(df[f"{roi}_z_dFF"], errors="coerce").to_numpy(dtype=float)
        label = "z-dFF"
    elif f"{roi}_dFF" in df.columns:
        y = pd.to_numeric(df[f"{roi}_dFF"], errors="coerce").to_numpy(dtype=float)
        sd = np.nanstd(y, ddof=1)
        z = (y - np.nanmean(y)) / sd if sd and np.isfinite(sd) else np.full_like(y, np.nan)
        label = "z-score of dFF"
    else:
        raise ValueError("Could not find z-dFF, dFF, or Delta-F columns for the z-score graph.")
    return z, label


def sliding_window_mean_by_time(time_sec, values, window_sec=1800.0, centered=True):
    """
    Time-aware sliding mean.

    Default window_sec = 1800 seconds = 30 minutes.
    """
    time_sec = np.asarray(time_sec, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(time_sec) == 0:
        return values

    window_sec = max(float(window_sec), 1.0)
    order = np.argsort(time_sec)
    sorted_time = time_sec[order]
    sorted_values = values[order]

    s = pd.Series(
        sorted_values,
        index=pd.to_timedelta(sorted_time, unit="s"),
    )

    smooth = (
        s.rolling(
            window=pd.Timedelta(seconds=window_sec),
            center=bool(centered),
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    out = np.empty_like(smooth, dtype=float)
    out[order] = smooth
    return out


def get_zscore_and_smooth(df, roi, source="z-dFF column", window_sec=1800.0, centered=True):
    """Return raw z-score and sliding-window mean z-score traces."""
    z, source_label = get_zscore_values(df, roi, source=source)
    smooth = sliding_window_mean_by_time(
        df["time_sec"].to_numpy(dtype=float),
        z,
        window_sec=window_sec,
        centered=centered,
    )
    return z, smooth, source_label


def add_repeating_dark_shading(fig, df, dark_start_sec=43200.0, dark_duration_sec=43200.0, cycle_sec=86400.0, rows=(1,)):
    """
    Optional gray shading for repeated dark-cycle epochs.
    Uses time_sec x-axis values, not decimal hours.
    """
    if df is None or df.empty or "time_sec" not in df.columns:
        return fig

    x_min = float(np.nanmin(df["time_sec"]))
    x_max = float(np.nanmax(df["time_sec"]))
    dark_start_sec = float(dark_start_sec)
    dark_duration_sec = max(float(dark_duration_sec), 1.0)
    cycle_sec = max(float(cycle_sec), dark_duration_sec + 1.0)

    first_start = dark_start_sec - np.ceil((dark_start_sec - x_min) / cycle_sec) * cycle_sec
    epoch_start = first_start

    while epoch_start <= x_max + cycle_sec:
        epoch_end = epoch_start + dark_duration_sec
        shade_start = max(epoch_start, x_min)
        shade_end = min(epoch_end, x_max)
        if shade_end > shade_start:
            for row in rows:
                fig.add_vrect(
                    x0=shade_start,
                    x1=shade_end,
                    fillcolor="rgba(120,120,120,0.16)",
                    line_width=0,
                    row=row,
                    col=1,
                )
        epoch_start += cycle_sec
    return fig


def make_zscore_sliding_mean_figure(
    df,
    roi,
    limits=None,
    events=None,
    exclusions=None,
    show_exclusions=True,
    show_event_labels=True,
    height=650,
    window_sec=1800.0,
    centered=True,
    source="z-dFF column",
    show_raw=True,
    show_smooth=True,
    show_dark_shading=False,
    dark_start_sec=43200.0,
    dark_duration_sec=43200.0,
    cycle_sec=86400.0,
):
    """
    Long-term z-score graph:
      black trace = raw z-score trace
      blue trace  = sliding-window mean z-score trace

    Default sliding mean window = 30 minutes.
    """
    z, smooth, source_label = get_zscore_and_smooth(
        df, roi, source=source, window_sec=window_sec, centered=centered
    )

    window_label = seconds_to_hms(window_sec)
    subtitle = f"Z-score with sliding mean window = {window_label}"

    fig = make_subplots(rows=1, cols=1, subplot_titles=(subtitle,))
    x = df["time_sec"]

    if show_dark_shading:
        add_repeating_dark_shading(
            fig,
            df,
            dark_start_sec=dark_start_sec,
            dark_duration_sec=dark_duration_sec,
            cycle_sec=cycle_sec,
            rows=(1,),
        )

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=z,
                mode="lines",
                name=f"Raw {source_label}",
                line=dict(color=graph_color("raw_z"), width=0.8),
                opacity=0.85,
            ),
            row=1,
            col=1,
        )

    if show_smooth:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=smooth,
                mode="lines",
                name=f"Sliding mean ({window_label})",
                line=dict(color=graph_color("z_smooth"), width=2.4),
            ),
            row=1,
            col=1,
        )

    fig.update_yaxes(title_text="Z score", row=1, col=1)
    apply_hms_xaxis(fig, df, rows=[1])

    if limits:
        apply_y_range(fig, 1, limits.get("z_dFF"))

    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1], show_labels=show_event_labels)

    return finish_fig(fig, height=height, df=df)


def summarize_zscore_ranges(df, z, smooth, periods, analysis_trace="Sliding mean"):
    """
    Calculate z-score summary statistics for multiple time windows.

    The graph display is visual-only; these calculations do not change the CSV.
    """
    rows = []
    time_sec = df["time_sec"].to_numpy(dtype=float)

    for idx, period in enumerate(periods, start=1):
        start = float(period["start_sec"])
        end = float(period["end_sec"])
        if end <= start:
            continue

        mask = (time_sec >= start) & (time_sec <= end)
        if not mask.any():
            rows.append({
                "Range": period.get("name", f"Range {idx}"),
                "Start": seconds_to_hms(start),
                "End": seconds_to_hms(end),
                "N samples": 0,
                "Trace used": analysis_trace,
                "Average z-score": np.nan,
                "Min z-score": np.nan,
                "Max z-score": np.nan,
                "Z-score range": np.nan,
            })
            continue

        values = z[mask] if analysis_trace == "Raw z-score" else smooth[mask]
        finite = values[np.isfinite(values)]

        if finite.size == 0:
            mean_val = min_val = max_val = range_val = np.nan
        else:
            mean_val = float(np.nanmean(finite))
            min_val = float(np.nanmin(finite))
            max_val = float(np.nanmax(finite))
            range_val = max_val - min_val

        rows.append({
            "Range": period.get("name", f"Range {idx}"),
            "Start": seconds_to_hms(start),
            "End": seconds_to_hms(end),
            "N samples": int(finite.size),
            "Trace used": analysis_trace,
            "Average z-score": mean_val,
            "Min z-score": min_val,
            "Max z-score": max_val,
            "Z-score range": range_val,
        })

    return pd.DataFrame(rows)


def add_zscore_range_shapes_and_stats(
    fig,
    df,
    z,
    smooth,
    periods,
    analysis_trace="Sliding mean",
    show_average_line=True,
    show_annotations=True,
):
    """Add range shading, boundary lines, and average z-score line segments."""
    if not periods:
        return fig, pd.DataFrame()

    stats_df = summarize_zscore_ranges(df, z, smooth, periods, analysis_trace=analysis_trace)

    for idx, period in enumerate(periods, start=1):
        start = float(period["start_sec"])
        end = float(period["end_sec"])
        if end <= start:
            continue

        color = period.get("color", graph_color("range_average"))
        shade_alpha = float(period.get("shade_alpha", 0.18))
        line_alpha = float(period.get("line_alpha", 0.75))
        line_color = hex_to_rgba(color, line_alpha)

        fig.add_vrect(x0=start, x1=end, fillcolor=color, opacity=shade_alpha, line_width=0, row=1, col=1)
        fig.add_vline(x=start, line_width=1.3, line_dash="dash", line_color=line_color, row=1, col=1)
        fig.add_vline(x=end, line_width=1.3, line_dash="dash", line_color=line_color, row=1, col=1)

        if idx - 1 < len(stats_df):
            row = stats_df.iloc[idx - 1]
            avg = row["Average z-score"]
            min_z = row["Min z-score"]
            max_z = row["Max z-score"]
            z_range = row["Z-score range"]

            if np.isfinite(avg) and show_average_line:
                fig.add_trace(
                    go.Scatter(
                        x=[start, end],
                        y=[avg, avg],
                        mode="lines",
                        name=f"{period.get('name', f'Range {idx}')} average z",
                        line=dict(color=color, width=3, dash="solid"),
                        hovertemplate=(
                            f"{period.get('name', f'Range {idx}')}<br>"
                            f"Average z-score: {avg:.4g}<br>"
                            f"Start: {seconds_to_hms(start)}<br>"
                            f"End: {seconds_to_hms(end)}<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

            if show_annotations and np.isfinite(avg):
                label = (
                    f"<b>{period.get('name', f'Range {idx}')}</b><br>"
                    f"Avg z = {avg:.3g}<br>"
                    f"Min = {min_z:.3g}, Max = {max_z:.3g}<br>"
                    f"Range = {z_range:.3g}"
                )
                fig.add_annotation(
                    x=(start + end) / 2,
                    y=avg,
                    xref="x",
                    yref="y",
                    text=label,
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-45,
                    font=dict(size=11, color="#111827"),
                    bgcolor="rgba(255,255,255,0.88)",
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=4,
                )

    return fig, stats_df


def make_zscore_range_analysis_figure(
    df,
    roi,
    limits=None,
    events=None,
    exclusions=None,
    show_exclusions=True,
    show_event_labels=True,
    height=700,
    window_sec=1800.0,
    centered=True,
    source="z-dFF column",
    show_raw=True,
    show_smooth=True,
    show_dark_shading=False,
    dark_start_sec=43200.0,
    dark_duration_sec=43200.0,
    cycle_sec=86400.0,
    range_periods=None,
    range_trace="Sliding mean",
    show_average_line=True,
    show_range_annotations=True,
):
    """
    Z-score graph with user-defined time windows.

    It uses the same z-score source and sliding-window size as the main z-score graph.
    For each selected range, the app calculates average z-score, min z-score,
    max z-score, and max-min range.
    """
    z, smooth, source_label = get_zscore_and_smooth(
        df, roi, source=source, window_sec=window_sec, centered=centered
    )

    window_label = seconds_to_hms(window_sec)
    subtitle = f"Z-score range analysis; sliding mean window = {window_label}"

    fig = make_subplots(rows=1, cols=1, subplot_titles=(subtitle,))
    x = df["time_sec"]

    if show_dark_shading:
        add_repeating_dark_shading(
            fig,
            df,
            dark_start_sec=dark_start_sec,
            dark_duration_sec=dark_duration_sec,
            cycle_sec=cycle_sec,
            rows=(1,),
        )

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=z,
                mode="lines",
                name=f"Raw {source_label}",
                line=dict(color=graph_color("raw_z"), width=0.8),
                opacity=0.65,
            ),
            row=1,
            col=1,
        )

    if show_smooth:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=smooth,
                mode="lines",
                name=f"Sliding mean ({window_label})",
                line=dict(color=graph_color("z_smooth"), width=2.4),
            ),
            row=1,
            col=1,
        )

    fig, stats_df = add_zscore_range_shapes_and_stats(
        fig,
        df,
        z,
        smooth,
        range_periods or [],
        analysis_trace=range_trace,
        show_average_line=show_average_line,
        show_annotations=show_range_annotations,
    )

    fig.update_yaxes(title_text="Z score", row=1, col=1)
    apply_hms_xaxis(fig, df, rows=[1])

    if limits:
        apply_y_range(fig, 1, limits.get("z_dFF"))

    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1], show_labels=show_event_labels)

    return finish_fig(fig, height=height, df=df), stats_df


def filter_events_for_context(events, recording_name, graph_kind):
    """
    Filter event annotations so events can be applied only to selected recordings
    and/or selected graph tabs.
    """
    filtered = []
    for ev in events or []:
        target_recordings = ev.get("target_recordings", ["All recordings"])
        target_graphs = ev.get("target_graphs", ["All graphs"])
        recording_ok = (not target_recordings or "All recordings" in target_recordings or recording_name in target_recordings)
        graph_ok = (not target_graphs or "All graphs" in target_graphs or graph_kind in target_graphs)
        if recording_ok and graph_ok:
            filtered.append(ev)
    return filtered


def render_zscore_display_controls(base_key):
    """Z-score controls shown inside the z-score graph tab/focus window."""
    st.markdown("#### Z-score display settings")
    c1, c2, c3 = st.columns([1.2, 1.0, 1.0])
    with c1:
        z_source = st.selectbox(
            "Trace source",
            ["z-dFF column", "Re-z-score dFF", "Re-z-score Delta-F"],
            index=0,
            key=f"{base_key}_z_source",
            help="Use the existing z-dFF column if available, or calculate a fresh z-score from dFF/Delta-F for display.",
        )
    with c2:
        z_window_hms = st.text_input(
            "Sliding mean window",
            value="00:30:00",
            key=f"{base_key}_z_window_hms",
            help="Use HH:MM:SS, for example 00:10:00, 00:30:00, or 01:00:00.",
        )
    with c3:
        z_centered_mean = st.checkbox(
            "Centered window",
            value=True,
            key=f"{base_key}_z_centered_mean",
            help="Centered smoothing avoids shifting peaks left/right. Turn off for a trailing running mean.",
        )
    z_window_sec, z_window_err = parse_hms_to_seconds(z_window_hms, 30 * 60, "sliding mean window")
    if z_window_err:
        st.error(z_window_err)
    c4, c5, c6 = st.columns(3)
    with c4:
        z_show_raw = st.checkbox("Show raw z-score trace", value=True, key=f"{base_key}_z_show_raw")
    with c5:
        z_show_smooth = st.checkbox("Show sliding mean trace", value=True, key=f"{base_key}_z_show_smooth")
    with c6:
        z_show_dark = st.checkbox("Show dark-cycle shading", value=False, key=f"{base_key}_z_show_dark")
    with st.expander("Dark-cycle shading settings", expanded=False):
        d1, d2, d3 = st.columns(3)
        with d1:
            z_dark_start_hms = st.text_input("Dark phase start", value="12:00:00", key=f"{base_key}_z_dark_start_hms")
        with d2:
            z_dark_duration_hms = st.text_input("Dark phase duration", value="12:00:00", key=f"{base_key}_z_dark_duration_hms")
        with d3:
            z_cycle_hms = st.text_input("Cycle length", value="24:00:00", key=f"{base_key}_z_cycle_hms")
        z_dark_start_sec, z_dark_start_err = parse_hms_to_seconds(z_dark_start_hms, 12 * 3600, "dark phase start")
        z_dark_duration_sec, z_dark_duration_err = parse_hms_to_seconds(z_dark_duration_hms, 12 * 3600, "dark phase duration")
        z_cycle_sec, z_cycle_err = parse_hms_to_seconds(z_cycle_hms, 24 * 3600, "cycle length")
        for err in [z_dark_start_err, z_dark_duration_err, z_cycle_err]:
            if err:
                st.error(err)
    return {
        "source": z_source,
        "window_sec": z_window_sec,
        "centered": z_centered_mean,
        "show_raw": z_show_raw,
        "show_smooth": z_show_smooth,
        "show_dark": z_show_dark,
        "dark_start_sec": z_dark_start_sec,
        "dark_duration_sec": z_dark_duration_sec,
        "cycle_sec": z_cycle_sec,
    }


def render_zscore_range_controls(base_key):
    """Z-score range controls shown inside the range-analysis tab/focus window."""
    st.markdown("#### Z-score range settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        z_range_trace = st.selectbox("Trace to summarize", ["Sliding mean", "Raw z-score"], index=0, key=f"{base_key}_z_range_trace")
    with c2:
        z_show_average_line = st.checkbox("Show average z-score line", value=True, key=f"{base_key}_z_show_average_line")
    with c3:
        z_show_range_annotations = st.checkbox("Show average/range labels", value=True, key=f"{base_key}_z_show_range_annotations")
    z_show_stats_table = st.checkbox("Show statistics table below graph", value=True, key=f"{base_key}_z_show_stats_table")
    if st.button("Clear z-score ranges", use_container_width=True, key=f"{base_key}_clear_z_ranges"):
        st.session_state[f"{base_key}_num_z_ranges"] = 0
    num_z_ranges = st.number_input("Number of z-score ranges", min_value=0, max_value=30, step=1, key=f"{base_key}_num_z_ranges")
    z_range_periods = []
    for i in range(int(num_z_ranges)):
        with st.container():
            st.markdown(f"**Z-score range {i + 1}**")
            c1, c2, c3 = st.columns([1.2, 1, 1])
            with c1:
                range_name = st.text_input("Range name", value=f"Range {i + 1}", key=f"{base_key}_z_range_name_{i}")
            with c2:
                range_start_hms = st.text_input("Start time", value="00:00:00", key=f"{base_key}_z_range_start_hms_{i}")
            with c3:
                range_end_hms = st.text_input("End time", value="00:30:00", key=f"{base_key}_z_range_end_hms_{i}")
            range_start_sec, range_start_err = parse_hms_to_seconds(range_start_hms, 0.0, f"z-score range {i + 1} start")
            range_end_sec, range_end_err = parse_hms_to_seconds(range_end_hms, 30 * 60, f"z-score range {i + 1} end")
            if range_start_err:
                st.error(range_start_err)
            if range_end_err:
                st.error(range_end_err)
            if range_end_sec <= range_start_sec:
                st.warning("End time must be after start time for this z-score range.")
            with st.container():
                st.markdown("Range color")
                c4, c5, c6 = st.columns(3)
                with c4:
                    range_color = st.color_picker("Range color", value="#f59e0b", key=f"{base_key}_z_range_color_{i}")
                with c5:
                    range_shade_alpha = st.slider("Shade transparency", 0.02, 0.80, 0.18, 0.02, key=f"{base_key}_z_range_shade_alpha_{i}")
                with c6:
                    range_line_alpha = st.slider("Boundary transparency", 0.05, 1.0, 0.75, 0.05, key=f"{base_key}_z_range_line_alpha_{i}")
            if range_end_sec > range_start_sec:
                z_range_periods.append({
                    "name": range_name,
                    "start_sec": float(range_start_sec),
                    "end_sec": float(range_end_sec),
                    "start_hhmmss": seconds_to_hms(range_start_sec),
                    "end_hhmmss": seconds_to_hms(range_end_sec),
                    "color": range_color,
                    "shade_alpha": float(range_shade_alpha),
                    "line_alpha": float(range_line_alpha),
                })
    return {
        "range_trace": z_range_trace,
        "show_average_line": z_show_average_line,
        "show_range_annotations": z_show_range_annotations,
        "show_stats_table": z_show_stats_table,
        "range_periods": z_range_periods,
    }


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

if "results" not in st.session_state:
    st.session_state["results"] = []
if "num_exclusions" not in st.session_state:
    st.session_state["num_exclusions"] = 0
if "num_events" not in st.session_state:
    st.session_state["num_events"] = 0

GRAPH_TARGET_OPTIONS = [
    "All graphs",
    "Overview",
    "Raw channels",
    "Main dFF",
    "Z-score sliding mean",
    "Z-score range analysis",
]

recording_target_options = ["All recordings"] + [
    str(r.get("name", f"Recording {i+1}"))
    for i, r in enumerate(st.session_state.get("results", []))
]

with st.sidebar:
    with st.expander("Inputs", expanded=True):
        uploaded_files = st.file_uploader(
            "Upload .ppd or .csv file(s)",
            type=["ppd", "csv"],
            accept_multiple_files=True,
            help="PPD files are sent to ana.py. CSV files are either processed directly or displayed if already processed.",
        )
        roi_name = st.text_input("ROI name", value="BLA")
        scale_mode = st.selectbox("Auto y-axis scaling", ["full", "robust", "none"], index=0)
        save_full = st.checkbox("Save full 20 Hz processed CSV for PPD runs", value=False)

    with st.expander("Graph display", expanded=False):
        graph_size = st.selectbox("Graph size", ["Normal", "Large", "Maximized"], index=0)
        focus_mode = st.checkbox("Maximize one graph only", value=False)
        focus_graph = st.selectbox(
            "Graph to maximize",
            ["Overview", "Raw channels", "Main dFF", "Z-score sliding mean", "Z-score range analysis"],
            index=0,
            disabled=not focus_mode,
        )
        show_event_labels = st.checkbox("Show event labels above graphs", value=True)
        show_hidden_region_shading = st.checkbox("Mark hidden regions with pale gray shading", value=True)
        st.caption("Mouse/trackpad scroll zoom is disabled. Use graph size/focus mode to view graphs larger.")

    with st.expander("Graph colors", expanded=False):
        st.caption("These colors apply across all graph tabs unless an event/range has its own color.")
        st.color_picker("465 raw / signal", value=DEFAULT_GRAPH_COLORS["sig_raw"], key="color_sig_raw")
        st.color_picker("405 raw / isosbestic", value=DEFAULT_GRAPH_COLORS["uv_raw"], key="color_uv_raw")
        st.color_picker("Fitted 405", value=DEFAULT_GRAPH_COLORS["uv_fit"], key="color_uv_fit")
        st.color_picker("Delta-F", value=DEFAULT_GRAPH_COLORS["deltaF"], key="color_deltaF")
        st.color_picker("dFF", value=DEFAULT_GRAPH_COLORS["dff"], key="color_dff")
        st.color_picker("Raw z-score", value=DEFAULT_GRAPH_COLORS["raw_z"], key="color_raw_z")
        st.color_picker("Sliding mean z-score", value=DEFAULT_GRAPH_COLORS["z_smooth"], key="color_z_smooth")
        st.color_picker("Default range average line", value=DEFAULT_GRAPH_COLORS["range_average"], key="color_range_average")

    with st.expander("Visual crop window", expanded=False):
        use_window = st.checkbox(
            "View only a time window",
            value=False,
            help="This only changes what you see on the webpage. It does not change the saved processed CSV.",
        )
        window_start_hms = st.text_input("Window start time (HH:MM:SS)", value="00:00:00")
        window_end_hms = st.text_input("Window end time (HH:MM:SS)", value="07:30:00")
        start_sec, start_err = parse_hms_to_seconds(window_start_hms, 0.0, "window start time")
        end_sec, end_err = parse_hms_to_seconds(window_end_hms, 7.5 * 3600, "window end time")
        if start_err:
            st.error(start_err)
        if end_err:
            st.error(end_err)
        if use_window and end_sec <= start_sec:
            st.error("Window end time must be after start time.")

    with st.expander("Crop recording and re-zero time", expanded=False):
        st.caption(
            "Unlike the visual crop above, this changes the data itself. "
            "The crop start becomes 00:00:00 and downloads match the graphs."
        )
        crop_enabled = st.checkbox("Crop and re-zero", value=False, key="crop_enabled")
        crop_start_hms = st.text_input(
            "Crop start (HH:MM:SS)", value="24:00:00", key="crop_start",
            help="Everything before this is discarded. This instant becomes 00:00:00.",
        )
        crop_end_hms = st.text_input(
            "Crop end (HH:MM:SS, blank = end of recording)", value="", key="crop_end",
        )
        crop_start_sec, crop_start_err = parse_hms_to_seconds(crop_start_hms, 0.0, "crop start")
        if crop_end_hms.strip():
            crop_end_sec, crop_end_err = parse_hms_to_seconds(crop_end_hms, 0.0, "crop end")
        else:
            crop_end_sec, crop_end_err = None, ""
        if crop_start_err:
            st.error(crop_start_err)
        if crop_end_err:
            st.error(crop_end_err)
        if crop_enabled and crop_end_sec is not None and crop_end_sec <= crop_start_sec:
            st.error("Crop end must be after crop start.")
        crop_recompute = st.checkbox(
            "Recompute z-score on the cropped window", value=True, key="crop_recompute",
            help="On: z-score describes the kept window. Off: keeps whole-session statistics.",
        )
        if crop_enabled:
            st.success(f"{crop_start_hms} will be shown as 00:00:00.")

    with st.expander("Lickometer", expanded=False):
        lick_enabled = st.checkbox(
            "Enable lickometer analysis", value=False, key="lick_enabled",
            help="Opens a dedicated lickometer section below the photometry graphs.",
        )
        lick_source = "None"
        lick_file = None
        lick_digital_channel = "digital_1"
        lick_time_unit = "s"
        lick_cfg = dict(lk.LICK_DEFAULTS)
        if lick_enabled:
            lick_source = st.radio(
                "Lick data source",
                ["pyPhotometry digital input", "Upload lick CSV"],
                key="lick_source",
                help="Digital input shares the photometry clock, so no alignment is needed.",
            )
            if lick_source == "pyPhotometry digital input":
                lick_digital_channel = st.selectbox(
                    "Digital channel", ["digital_1", "digital_2"], key="lick_dig_ch"
                )
            else:
                lick_file = st.file_uploader(
                    "Lick CSV", type=["csv"], key="lick_csv",
                    help="Either one timestamp per lick, or a time column plus a 0/1 lick-state column.",
                )
                lick_time_unit = st.selectbox(
                    "Time unit in lick file", ["s", "ms", "min"], key="lick_unit"
                )

            st.markdown("**Bout detection**")
            lick_cfg["min_inter_lick_sec"] = st.number_input(
                "Debounce / minimum inter-lick interval (s)",
                0.0, 1.0, float(lk.LICK_DEFAULTS["min_inter_lick_sec"]), 0.01,
                key="lick_debounce",
            )
            lick_cfg["inter_bout_sec"] = st.number_input(
                "Gap that separates bouts (s)",
                0.1, 60.0, float(lk.LICK_DEFAULTS["inter_bout_sec"]), 0.1,
                key="lick_ibi",
            )
            lick_cfg["min_licks_per_bout"] = int(st.number_input(
                "Minimum licks per bout", 1, 100,
                int(lk.LICK_DEFAULTS["min_licks_per_bout"]), 1, key="lick_minlicks",
            ))
            lick_cfg["rate_window_sec"] = st.number_input(
                "Lick-rate window (s)", 1.0, 3600.0,
                float(lk.LICK_DEFAULTS["rate_window_sec"]), 1.0, key="lick_ratewin",
            )

            st.markdown("**Peri-event photometry**")
            lick_cfg["pre_sec"] = st.number_input(
                "Seconds before bout onset", 1.0, 300.0,
                float(lk.LICK_DEFAULTS["pre_sec"]), 1.0, key="lick_pre",
            )
            lick_cfg["post_sec"] = st.number_input(
                "Seconds after bout onset", 1.0, 600.0,
                float(lk.LICK_DEFAULTS["post_sec"]), 1.0, key="lick_post",
            )
            lick_cfg["baseline_start_sec"] = st.number_input(
                "Baseline window start (s, relative to onset)",
                -300.0, 0.0, float(lk.LICK_DEFAULTS["baseline_start_sec"]), 0.5,
                key="lick_bl_start",
            )
            lick_cfg["baseline_end_sec"] = st.number_input(
                "Baseline window end (s, relative to onset)",
                -300.0, 0.0, float(lk.LICK_DEFAULTS["baseline_end_sec"]), 0.5,
                key="lick_bl_end",
            )
            lick_norm = st.selectbox(
                "Peri-event normalisation",
                ["baseline_z", "baseline_sub", "none"], key="lick_norm",
                help="baseline_z expresses each trial in SDs of its own pre-onset baseline.",
            )
        else:
            lick_norm = "baseline_z"

    with st.expander("Hide/remove time periods visually", expanded=False):
        st.caption("These controls only affect graph display. They do not alter downloaded processed CSVs.")
        if st.button("Clear hidden time periods", use_container_width=True):
            st.session_state["num_exclusions"] = 0
        exclusion_mode = st.radio(
            "How to hide selected regions",
            ["Blank signal but keep true time axis", "Remove rows from visual display"],
            index=0,
        )
        exclusion_mode_code = "blank" if exclusion_mode.startswith("Blank") else "remove_rows"
        num_exclusions = st.number_input(
            "Number of time periods to hide",
            min_value=0,
            max_value=20,
            step=1,
            key="num_exclusions",
        )
        visual_exclusions = []
        for i in range(int(num_exclusions)):
            with st.container():
                st.markdown(f"**Hidden region {i + 1} timing**")
                ex_start_hms = st.text_input(f"Hidden region {i + 1} start (HH:MM:SS)", value="00:00:00", key=f"ex_start_hms_{i}")
                ex_end_hms = st.text_input(f"Hidden region {i + 1} end (HH:MM:SS)", value="00:01:00", key=f"ex_end_hms_{i}")
                ex_start, ex_start_err = parse_hms_to_seconds(ex_start_hms, 0.0, f"hidden region {i + 1} start")
                ex_end, ex_end_err = parse_hms_to_seconds(ex_end_hms, 60.0, f"hidden region {i + 1} end")
                if ex_start_err:
                    st.error(ex_start_err)
                if ex_end_err:
                    st.error(ex_end_err)
                if ex_end > ex_start:
                    visual_exclusions.append({
                        "start_sec": ex_start,
                        "end_sec": ex_end,
                        "start_hhmmss": seconds_to_hms(ex_start),
                        "end_hhmmss": seconds_to_hms(ex_end),
                    })
                else:
                    st.warning("End time must be after start time for this hidden region.")

    with st.expander("Event lines / shaded epochs", expanded=False):
        if st.button("Clear event annotations", use_container_width=True):
            st.session_state["num_events"] = 0
        num_events = st.number_input(
            "Number of events/epochs",
            min_value=0,
            max_value=50,
            step=1,
            key="num_events",
        )
        event_annotations = []
        for i in range(int(num_events)):
            with st.container():
                st.markdown(f"**Event / epoch {i + 1}: name and timing**")
                name = st.text_input("Event name", value=f"Event {i + 1}", key=f"ev_name_{i}")
                start_hms = st.text_input("Start time (HH:MM:SS)", value="00:00:00", key=f"ev_start_hms_{i}")
                start, start_time_err = parse_hms_to_seconds(start_hms, 0.0, f"event {i + 1} start time")
                if start_time_err:
                    st.error(start_time_err)
                has_duration = st.checkbox("This event lasts for a duration", value=False, key=f"ev_duration_{i}")
                end = None
                end_hms = ""
                if has_duration:
                    end_hms = st.text_input("End time (HH:MM:SS)", value=seconds_to_hms(start + 60.0), key=f"ev_end_hms_{i}")
                    end, end_time_err = parse_hms_to_seconds(end_hms, start + 60.0, f"event {i + 1} end time")
                    if end_time_err:
                        st.error(end_time_err)
                    if end <= start:
                        st.warning("Event end time must be after start time to shade a duration.")
                        end = None

            with st.container():
                st.markdown(f"**Event / epoch {i + 1}: apply to**")
                target_recordings = st.multiselect(
                    "Apply to recording(s)",
                    options=recording_target_options,
                    default=["All recordings"],
                    key=f"ev_target_recordings_{i}",
                    help="Use this if you uploaded multiple recordings and only want the event on one of them.",
                )
                target_graphs = st.multiselect(
                    "Apply to graph tab(s)",
                    options=GRAPH_TARGET_OPTIONS,
                    default=["All graphs"],
                    key=f"ev_target_graphs_{i}",
                    help="Use this if you only want the event line on selected graph tabs.",
                )

            with st.container():
                st.markdown(f"**Event / epoch {i + 1}: color and shading**")
                color = st.color_picker("Dotted line / shade color", value="#ff4b4b", key=f"ev_color_{i}")
                alpha = st.slider("Dotted line transparency", 0.05, 1.0, 0.65, 0.05, key=f"ev_alpha_{i}")
                shade_alpha = st.slider("Duration shade transparency", 0.02, 0.80, 0.18, 0.02, key=f"ev_shade_alpha_{i}")

            event_annotations.append({
                "name": name,
                "start_sec": float(start),
                "end_sec": float(end) if end is not None else None,
                "start_hhmmss": seconds_to_hms(start),
                "end_hhmmss": seconds_to_hms(end) if end is not None else None,
                "color": color,
                "alpha": float(alpha),
                "shade_alpha": float(shade_alpha),
                "target_recordings": target_recordings,
                "target_graphs": target_graphs,
            })

    with st.expander("Manual y-scales", expanded=False):
        st.caption("Leave blank to use autoscale. Type as: min,max")
        manual_raw = st.text_input("Raw y-scale", value="")
        manual_fit = st.text_input("Fit-check y-scale", value="")
        manual_delta = st.text_input("Delta-F y-scale", value="")
        manual_dff = st.text_input("dFF (%) y-scale", value="")
        manual_z = st.text_input("z-dFF y-scale", value="")

# =============================================================================
# RUN ANALYSIS
# =============================================================================

if "results" not in st.session_state:
    st.session_state["results"] = []
if "num_exclusions" not in st.session_state:
    st.session_state["num_exclusions"] = 0
if "num_events" not in st.session_state:
    st.session_state["num_events"] = 0
if "num_z_ranges" not in st.session_state:
    st.session_state["num_z_ranges"] = 0

run_clicked = st.button("Run / refresh analysis", type="primary", use_container_width=True)

if run_clicked:
    if not uploaded_files:
        st.warning("Upload at least one .ppd or .csv file first.")
    else:
        all_results = []
        with st.spinner("Running analysis..."):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                saved_files = []
                for f in uploaded_files:
                    path = tmp / safe_name(f.name)
                    path.write_bytes(f.getbuffer())
                    saved_files.append(path)

                ppd_files = [p for p in saved_files if p.suffix.lower() == ".ppd"]
                csv_files = [p for p in saved_files if p.suffix.lower() == ".csv"]

                # PPD files: run your existing ana.py
                if ppd_files:
                    ppd_out = tmp / "ppd_outputs"
                    command = [
                        sys.executable,
                        "ana.py",
                        *[str(p) for p in ppd_files],
                        "--roi",
                        roi_name,
                        "--outdir",
                        str(ppd_out),
                        "--scale-mode",
                        scale_mode,
                    ]
                    if save_full:
                        command.append("--save-full")
                    result = subprocess.run(command, capture_output=True, text=True)

                    if result.returncode != 0:
                        st.error("PPD analysis failed.")
                        st.code(result.stderr)
                    else:
                        st.success("PPD analysis complete.")
                        with st.expander("PPD run log"):
                            st.code(result.stdout)
                        processed_csvs = sorted(ppd_out.rglob("processed_jones_style_downsampled*.csv"))
                        if not processed_csvs:
                            processed_csvs = sorted(ppd_out.rglob("*.csv"))
                        for csv_path in processed_csvs:
                            try:
                                df = pd.read_csv(csv_path)
                                df = load_processed_csv_dataframe(df, roi_name)[0]
                                # Full-rate digital event times written alongside
                                # the processed CSV, so licking keeps its true
                                # timing even though photometry is exported at 1 Hz.
                                digital_events = {}
                                for ev_path in csv_path.parent.glob("event_times_digital_*.csv"):
                                    ch = ev_path.stem.replace("event_times_", "")
                                    try:
                                        digital_events[ch] = pd.read_csv(ev_path)["event_time_sec"].to_numpy(float)
                                    except Exception:
                                        pass
                                all_results.append({
                                    "name": csv_path.parent.name if csv_path.parent.name else csv_path.stem,
                                    "df": df,
                                    "settings": {"input_type": "ppd", "source_csv": str(csv_path)},
                                    "digital_events": digital_events,
                                })
                            except Exception as e:
                                st.warning(f"Could not load PPD output CSV {csv_path.name}: {e}")

                # CSV files: process or display directly
                for csv_path in csv_files:
                    try:
                        df_in = pd.read_csv(csv_path)
                        if processed_columns_exist(df_in, roi_name):
                            df_out, settings = load_processed_csv_dataframe(df_in, roi_name)
                        else:
                            df_out, settings = process_raw_csv_dataframe(df_in, roi_name)
                        all_results.append({"name": csv_path.stem, "df": df_out, "settings": settings})
                    except Exception as e:
                        st.error(f"CSV analysis failed for {csv_path.name}")
                        st.code(str(e))
        st.session_state["results"] = all_results

# =============================================================================
# DISPLAY RESULTS
# =============================================================================

results = st.session_state.get("results", [])

# Apply the physical crop before any figure, metric or download is produced, so
# the graphs, the summary numbers and the exported CSV all agree.
crop_offset_applied = 0.0
if results and st.session_state.get("crop_enabled"):
    _n_before = len(results)
    results = apply_crop_to_results(
        results, roi_name, crop_start_sec, crop_end_sec, recompute_stats=crop_recompute
    )
    crop_offset_applied = float(crop_start_sec or 0.0)
    event_annotations = shift_events_for_crop(event_annotations, crop_offset_applied)
    if not results:
        st.error(
            f"Crop from {crop_start_hms} removed all data from every recording. "
            "Check that the crop start is inside the recording."
        )
    elif len(results) < _n_before:
        st.warning(f"{_n_before - len(results)} recording(s) had no data inside the crop window.")


if not results:
    st.info("Upload a `.ppd` or `.csv` file, then click **Run / refresh analysis**.")
else:
    auto = get_auto_limits(results, roi_name)
    manual_limits = {
        "raw": parse_limit_text(manual_raw, auto.get("raw")),
        "fit": parse_limit_text(manual_fit, auto.get("fit")),
        "deltaF": parse_limit_text(manual_delta, auto.get("deltaF")),
        "dFF": parse_limit_text(manual_dff, auto.get("dFF")),
        "z_dFF": parse_limit_text(manual_z, auto.get("z_dFF")),
    }

    height_mult = {"Normal": 1.0, "Large": 1.35, "Maximized": 1.85}[graph_size]
    overview_h = int(850 * height_mult)
    raw_h = int(650 * height_mult)
    processed_h = int(760 * height_mult)
    z_h = int(650 * height_mult)
    z_range_h = int(720 * height_mult)
    dual_h = int(560 * height_mult)

    st.success(f"Loaded {len(results)} recording(s).")
    c1, c2, c3, c4 = st.columns(4)
    max_duration = max(float(r["df"]["time_sec"].max()) for r in results)
    c1.metric("Recordings loaded", len(results))
    c2.metric("Max duration", seconds_to_hms(max_duration))
    c3.metric("ROI", roi_name)
    c4.metric("Hidden regions", len(visual_exclusions))

    with st.expander("Current y-axis limits and visual window"):
        st.json({
            "auto_limits": {k: list(v) if v else None for k, v in auto.items()},
            "active_limits_after_manual_override": {k: list(v) if v else None for k, v in manual_limits.items()},
            "visual_window": [seconds_to_hms(start_sec), seconds_to_hms(end_sec)] if use_window else "full recording",
            "note": "Visual window affects only webpage display, not downloaded processed CSV files.",
        })

    st.markdown(
        """
        <div class="note-box-light">
        Scroll-wheel zoom is disabled on the graphs. Use <b>Graph display → Graph size</b> or <b>Maximize one graph only</b> for a larger view.
        Visual crop, hidden regions, and event annotations change only the webpage display, not the downloaded processed CSV.
        </div>
        """,
        unsafe_allow_html=True,
    )

    for idx, result in enumerate(results, start=1):
        full_df = result["df"]
        view_df = filter_visual_window(full_df, start_sec, end_sec) if use_window else full_df
        view_df = apply_visual_exclusions(view_df, roi_name, visual_exclusions, mode=exclusion_mode_code)
        if view_df.empty:
            st.warning(f"{result['name']}: visual window has no data.")
            continue

        st.markdown(f"## Recording {idx}: `{result['name']}`")

        def render_chart(kind):
            events_for_this_graph = filter_events_for_context(
                event_annotations,
                str(result["name"]),
                kind,
            )
            if kind == "Overview":
                fig = make_overview_figure(
                    view_df, roi_name, manual_limits,
                    events=events_for_this_graph,
                    exclusions=visual_exclusions,
                    show_exclusions=show_hidden_region_shading,
                    show_event_labels=show_event_labels,
                    height=overview_h,
                )
            elif kind == "Raw channels":
                fig = make_raw_independent_figure(
                    view_df, roi_name, manual_limits,
                    events=events_for_this_graph,
                    exclusions=visual_exclusions,
                    show_exclusions=show_hidden_region_shading,
                    show_event_labels=show_event_labels,
                    height=raw_h,
                )
            elif kind == "Z-score sliding mean":
                zcfg = render_zscore_display_controls(f"{safe_name(result['name'])}_{idx}_zslide")
                fig = make_zscore_sliding_mean_figure(
                    view_df,
                    roi_name,
                    manual_limits,
                    events=events_for_this_graph,
                    exclusions=visual_exclusions,
                    show_exclusions=show_hidden_region_shading,
                    show_event_labels=show_event_labels,
                    height=z_h,
                    window_sec=zcfg["window_sec"],
                    centered=zcfg["centered"],
                    source=zcfg["source"],
                    show_raw=zcfg["show_raw"],
                    show_smooth=zcfg["show_smooth"],
                    show_dark_shading=zcfg["show_dark"],
                    dark_start_sec=zcfg["dark_start_sec"],
                    dark_duration_sec=zcfg["dark_duration_sec"],
                    cycle_sec=zcfg["cycle_sec"],
                )
            elif kind == "Z-score range analysis":
                zcfg = render_zscore_display_controls(f"{safe_name(result['name'])}_{idx}_zrange")
                zrange_cfg = render_zscore_range_controls(f"{safe_name(result['name'])}_{idx}")
                fig, z_stats_df = make_zscore_range_analysis_figure(
                    view_df,
                    roi_name,
                    manual_limits,
                    events=events_for_this_graph,
                    exclusions=visual_exclusions,
                    show_exclusions=show_hidden_region_shading,
                    show_event_labels=show_event_labels,
                    height=z_range_h,
                    window_sec=zcfg["window_sec"],
                    centered=zcfg["centered"],
                    source=zcfg["source"],
                    show_raw=zcfg["show_raw"],
                    show_smooth=zcfg["show_smooth"],
                    show_dark_shading=zcfg["show_dark"],
                    dark_start_sec=zcfg["dark_start_sec"],
                    dark_duration_sec=zcfg["dark_duration_sec"],
                    cycle_sec=zcfg["cycle_sec"],
                    range_periods=zrange_cfg["range_periods"],
                    range_trace=zrange_cfg["range_trace"],
                    show_average_line=zrange_cfg["show_average_line"],
                    show_range_annotations=zrange_cfg["show_range_annotations"],
                )
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                if zrange_cfg["show_stats_table"] and z_stats_df is not None and not z_stats_df.empty:
                    st.dataframe(
                        z_stats_df.style.format({
                            "Average z-score": "{:.4f}",
                            "Min z-score": "{:.4f}",
                            "Max z-score": "{:.4f}",
                            "Z-score range": "{:.4f}",
                        }),
                        use_container_width=True,
                        height=min(420, 105 + 35 * len(z_stats_df)),
                    )
                    st.download_button(
                        label="Download z-score range statistics as CSV",
                        data=z_stats_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{safe_name(result['name'])}_zscore_range_statistics.csv",
                        mime="text/csv",
                    )
                return
            else:
                fig = make_main_dff_figure(
                    view_df, roi_name, manual_limits,
                    events=events_for_this_graph,
                    exclusions=visual_exclusions,
                    show_exclusions=show_hidden_region_shading,
                    show_event_labels=show_event_labels,
                    height=processed_h,
                )
            st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)

        if focus_mode:
            st.markdown(f"### Maximized view: {focus_graph}")
            render_chart(focus_graph)
            with st.expander("Data preview and downloads"):
                st.caption("Preview of the displayed data window. The downloadable CSV remains the full processed data.")
                st.dataframe(view_df.head(500), use_container_width=True, height=360)
                st.download_button(
                    label=f"Download full processed CSV for {result['name']}",
                    data=full_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{safe_name(result['name'])}_processed_output.csv",
                    mime="text/csv",
                )
            continue

        tabs = st.tabs(["Overview", "Raw channels", "Main dFF", "Z-score sliding mean", "Z-score range analysis", "Data preview", "Settings"])

        with tabs[0]:
            render_chart("Overview")
        with tabs[1]:
            render_chart("Raw channels")
        with tabs[2]:
            render_chart("Main dFF")
        with tabs[3]:
            render_chart("Z-score sliding mean")
        with tabs[4]:
            render_chart("Z-score range analysis")
        with tabs[5]:
            st.caption("Preview of the displayed data window. The downloadable CSV remains the full processed data.")
            st.dataframe(view_df.head(500), use_container_width=True, height=360)
            st.download_button(
                label=f"Download full processed CSV for {result['name']}",
                data=full_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_name(result['name'])}_processed_output.csv",
                mime="text/csv",
            )
        with tabs[6]:
            st.json(result.get("settings", {}))

    # =========================================================================
    # LICKOMETER SECTION
    # Rendered only when enabled in the sidebar, so the photometry workflow is
    # unchanged for anyone not recording licks.
    # =========================================================================
    if st.session_state.get("lick_enabled"):
        st.divider()
        st.markdown("## Lickometer")

        lick_times_by_rec = {}
        lick_provenance = {}

        # ---- resolve lick times for each recording --------------------------
        if lick_source == "Upload lick CSV" and lick_file is not None:
            try:
                lick_df_in = pd.read_csv(lick_file)
                cols = list(lick_df_in.columns)
                c1, c2 = st.columns(2)
                with c1:
                    lick_mode = st.radio(
                        "Lick CSV layout",
                        ["One timestamp per lick", "Time column + 0/1 state column"],
                        key="lick_csv_mode",
                    )
                if lick_mode == "One timestamp per lick":
                    with c2:
                        tcol = st.selectbox("Lick time column", cols, key="lick_tcol")
                    times, _ = lk.licks_from_timestamp_csv(
                        lick_df_in, time_column=tcol, unit=lick_time_unit,
                        min_inter_lick_sec=lick_cfg["min_inter_lick_sec"],
                    )
                else:
                    with c2:
                        tcol = st.selectbox("Time column", cols, key="lick_tcol2")
                        scol = st.selectbox("Lick state column", cols, key="lick_scol")
                    times = lk.licks_from_state_csv(
                        lick_df_in, tcol, scol, unit=lick_time_unit,
                        min_inter_lick_sec=lick_cfg["min_inter_lick_sec"],
                    )
                if crop_offset_applied:
                    times = times - crop_offset_applied
                    st.caption(
                        f"Lick times shifted by -{seconds_to_hms(crop_offset_applied)} "
                        "to match the cropped photometry timeline."
                    )
                for r in results:
                    lick_times_by_rec[r["name"]] = times
            except Exception as e:
                st.error(f"Could not read the lick CSV: {e}")

        elif lick_source == "pyPhotometry digital input":
            for r in results:
                d = r["df"]
                times = None
                provenance = ""

                # 1. Best: exact rising-edge times captured at the acquisition
                #    rate before the photometry was downsampled.
                ev = (r.get("digital_events") or {}).get(lick_digital_channel)
                if ev is not None and len(ev):
                    times = np.asarray(ev, dtype=float)
                    provenance = "full-rate digital timestamps"

                # 2. Fallback: reconstruct from per-bin pulse counts. A 1 Hz
                #    export cannot place licks within a bin, so they are spread
                #    evenly across it. Counts and bout structure are preserved;
                #    individual inter-lick intervals are approximate.
                elif f"{lick_digital_channel}_pulse_count" in d.columns:
                    t_arr = d["time_sec"].to_numpy(dtype=float)
                    counts = pd.to_numeric(
                        d[f"{lick_digital_channel}_pulse_count"], errors="coerce"
                    ).fillna(0).to_numpy(int)
                    dt = float(np.median(np.diff(t_arr))) if t_arr.size > 1 else 1.0
                    rebuilt = []
                    for t0, c in zip(t_arr, counts):
                        if c > 0:
                            rebuilt.append(t0 + (np.arange(c) + 0.5) * (dt / c))
                    times = np.concatenate(rebuilt) if rebuilt else np.array([])
                    provenance = "reconstructed from per-bin pulse counts (approximate timing)"

                # 3. Last resort: rising edges of a binary column.
                elif lick_digital_channel in d.columns:
                    t_arr = d["time_sec"].to_numpy(dtype=float)
                    dig = pd.to_numeric(d[lick_digital_channel], errors="coerce").fillna(0).to_numpy()
                    rising = np.flatnonzero(np.diff((dig > 0).astype(np.int8)) == 1) + 1
                    times = t_arr[rising]
                    provenance = "rising edges of a downsampled binary column"

                if times is None:
                    continue
                if lick_cfg["min_inter_lick_sec"] > 0 and times.size:
                    keep = [times[0]]
                    for tv in times[1:]:
                        if tv - keep[-1] >= lick_cfg["min_inter_lick_sec"]:
                            keep.append(tv)
                    times = np.asarray(keep)
                lick_times_by_rec[r["name"]] = times
                lick_provenance[r["name"]] = provenance

            if not lick_times_by_rec:
                th.note(
                    f"No <b>{lick_digital_channel}</b> column found in the loaded data. "
                    "Re-run the analysis so the digital inputs are exported, or switch "
                    "to <b>Upload lick CSV</b>.",
                    kind="warn",
                )

        # ---- render per recording -------------------------------------------
        for r in results:
            times = np.asarray(lick_times_by_rec.get(r["name"], []), dtype=float)
            d = r["df"]
            st.markdown(f"### `{r['name']}`")

            if times.size == 0:
                st.info("No licks available for this recording.")
                continue

            duration = float(d["time_sec"].max() - d["time_sec"].min())
            bouts = lk.detect_bouts(
                times,
                inter_bout_sec=lick_cfg["inter_bout_sec"],
                min_licks_per_bout=lick_cfg["min_licks_per_bout"],
            )
            summary = lk.bout_summary_table(bouts, times, duration)

            prov = lick_provenance.get(r["name"], "")
            if "approximate" in prov:
                th.note(
                    f"Lick times {prov}. Counts and bout structure are reliable; "
                    "individual inter-lick intervals are not. Re-run the .ppd analysis "
                    "to get exact timestamps.",
                    kind="warn",
                )
            elif "rising edges of a downsampled" in prov:
                th.note(
                    "Lick times came from a downsampled binary column, so licks within "
                    "one bin were merged and counts will be underestimated.",
                    kind="warn",
                )

            mcols = st.columns(len(summary.columns))
            for col, name in zip(mcols, summary.columns):
                col.metric(name, summary.iloc[0][name])

            ltabs = st.tabs(["Licking over time", "Peri-bout photometry", "Bout table"])

            # -------- licking over time, aligned under dF/F -------------------
            with ltabs[0]:
                grid = d["time_sec"].to_numpy(dtype=float)
                rate = lk.lick_rate_trace(times, grid, window_sec=lick_cfg["rate_window_sec"])
                dff_col = f"{roi_name}_z_dFF" if f"{roi_name}_z_dFF" in d.columns else f"{roi_name}_dFF"

                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.44, 0.28, 0.28], vertical_spacing=0.05,
                    subplot_titles=(
                        f"{roi_name} {'z-dF/F' if dff_col.endswith('z_dFF') else 'dF/F'}",
                        f"Lick rate ({lick_cfg['rate_window_sec']:g} s window)",
                        "Lick raster",
                    ),
                )
                if dff_col in d.columns:
                    fig.add_trace(go.Scatter(
                        x=grid, y=pd.to_numeric(d[dff_col], errors="coerce"),
                        mode="lines", name=dff_col,
                        line=dict(width=1.1, color=th.TRACE_COLORS["dff"]),
                    ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=grid, y=rate, mode="lines", name="Lick rate (Hz)",
                    line=dict(width=1.2, color=th.TRACE_COLORS["lick_rate"]),
                    fill="tozeroy", fillcolor="rgba(8,145,178,0.15)",
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=times, y=np.ones_like(times), mode="markers", name="Licks",
                    marker=dict(symbol="line-ns-open", size=9, line=dict(width=1),
                                color=th.TRACE_COLORS["lick"]),
                ), row=3, col=1)

                for _, b in bouts.iterrows():
                    fig.add_vrect(
                        x0=b.onset_sec, x1=b.offset_sec,
                        fillcolor="rgba(245,158,11,0.14)", line_width=0, row=1, col=1,
                    )

                fig.update_yaxes(title_text="Hz", row=2, col=1)
                fig.update_yaxes(showticklabels=False, range=[0.5, 1.5], row=3, col=1)
                th.style_figure(fig, height=int(720 * height_mult))
                apply_hms_xaxis(fig, d, rows=[1, 2, 3])
                st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)
                st.caption("Shaded bands mark detected licking bouts.")

            # -------- peri-bout photometry ------------------------------------
            with ltabs[1]:
                dff_col = f"{roi_name}_dFF" if f"{roi_name}_dFF" in d.columns else None
                if dff_col is None or bouts.empty:
                    st.info("Need dF/F and at least one detected bout.")
                else:
                    mat, taxis, kept = lk.peri_event_matrix(
                        d["time_sec"].to_numpy(dtype=float),
                        pd.to_numeric(d[dff_col], errors="coerce").to_numpy(float),
                        bouts.onset_sec.to_numpy(float),
                        pre_sec=lick_cfg["pre_sec"], post_sec=lick_cfg["post_sec"],
                        baseline_start_sec=lick_cfg["baseline_start_sec"],
                        baseline_end_sec=lick_cfg["baseline_end_sec"],
                        normalise=lick_norm,
                    )
                    if mat.size == 0:
                        st.warning(
                            "No bout had a complete window inside the recording. "
                            "Shorten the pre/post window."
                        )
                    else:
                        dropped = len(bouts) - len(kept)
                        if dropped:
                            st.caption(
                                f"{dropped} bout(s) excluded: window extended past the "
                                "recording edge. Partial windows are never zero-padded."
                            )
                        unit = ("SD of baseline" if lick_norm == "baseline_z"
                                else "dF/F" if lick_norm != "none" else "dF/F (raw)")
                        mean = np.nanmean(mat, axis=0)
                        sem = np.nanstd(mat, axis=0, ddof=1) / np.sqrt(mat.shape[0]) \
                            if mat.shape[0] > 1 else np.zeros_like(mean)

                        pf = make_subplots(
                            rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.45, 0.55], vertical_spacing=0.07,
                            subplot_titles=(
                                f"Mean \u00b1 SEM across {mat.shape[0]} bouts",
                                "Per-bout heatmap",
                            ),
                        )
                        pf.add_trace(go.Scatter(
                            x=np.concatenate([taxis, taxis[::-1]]),
                            y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                            fill="toself", fillcolor="rgba(213,94,0,0.20)",
                            line=dict(width=0), hoverinfo="skip", showlegend=False,
                        ), row=1, col=1)
                        pf.add_trace(go.Scatter(
                            x=taxis, y=mean, mode="lines", name="Mean",
                            line=dict(width=2, color=th.TRACE_COLORS["dff"]),
                        ), row=1, col=1)
                        pf.add_trace(go.Heatmap(
                            z=mat, x=taxis, y=np.arange(1, mat.shape[0] + 1),
                            colorscale="RdBu_r", zmid=0,
                            colorbar=dict(title=unit, len=0.5, y=0.22),
                        ), row=2, col=1)
                        for rr in (1, 2):
                            pf.add_vline(x=0, line=dict(color="#111827", width=1.4,
                                                        dash="dash"), row=rr, col=1)
                        pf.update_xaxes(title_text="Time from bout onset (s)", row=2, col=1)
                        pf.update_yaxes(title_text=unit, row=1, col=1)
                        pf.update_yaxes(title_text="Bout", row=2, col=1)
                        th.style_figure(pf, height=int(660 * height_mult))
                        st.plotly_chart(pf, use_container_width=True, config=PLOTLY_CONFIG)

                        bl = (taxis >= lick_cfg["baseline_start_sec"]) & (taxis <= lick_cfg["baseline_end_sec"])
                        pk = (taxis >= 0) & (taxis <= min(10.0, lick_cfg["post_sec"]))
                        s1, s2, s3 = st.columns(3)
                        s1.metric("Bouts analysed", mat.shape[0])
                        s2.metric("Mean baseline", f"{np.nanmean(mat[:, bl]):.3f}")
                        s3.metric("Mean 0-10 s post-onset", f"{np.nanmean(mat[:, pk]):.3f}")

                        out_df = pd.DataFrame(mat.T, columns=[f"bout_{i+1}" for i in range(mat.shape[0])])
                        out_df.insert(0, "time_from_onset_sec", taxis)
                        st.download_button(
                            "Download peri-bout matrix (CSV)",
                            out_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{safe_name(r['name'])}_peri_bout.csv",
                            mime="text/csv",
                        )

            # -------- bout table ----------------------------------------------
            with ltabs[2]:
                show = bouts.copy()
                for c in ("onset_sec", "offset_sec"):
                    show[c.replace("_sec", "_hms")] = [seconds_to_hms(v) for v in show[c]]
                st.dataframe(show.round(3), use_container_width=True, height=340)
                st.download_button(
                    "Download bout table (CSV)",
                    show.to_csv(index=False).encode("utf-8"),
                    file_name=f"{safe_name(r['name'])}_lick_bouts.csv",
                    mime="text/csv",
                )
                st.download_button(
                    "Download lick times (CSV)",
                    pd.DataFrame({"lick_time_sec": times}).to_csv(index=False).encode("utf-8"),
                    file_name=f"{safe_name(r['name'])}_lick_times.csv",
                    mime="text/csv",
                )

    st.divider()
    st.download_button(
        label="Download all full processed outputs as ZIP",
        data=make_zip_from_results(results, roi_name),
        file_name="photometry_analysis_outputs.zip",
        mime="application/zip",
        use_container_width=True,
    )
