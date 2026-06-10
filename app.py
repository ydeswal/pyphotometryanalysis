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

# =============================================================================
# PAGE STYLE: SMALL HEADER + WIDE GRAPH WORKSPACE
# =============================================================================

st.set_page_config(
    page_title="Photometry Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.7rem !important;
        padding-left: 1.4rem !important;
        padding-right: 1.4rem !important;
        max-width: 100% !important;
    }

    .small-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background: rgba(14, 17, 23, 0.96);
        border-bottom: 1px solid rgba(255,255,255,0.12);
        padding: 0.35rem 0.25rem 0.45rem 0.25rem;
        margin-bottom: 0.6rem;
    }

    .small-header-title {
        font-size: 1.05rem;
        font-weight: 700;
        line-height: 1.1;
        margin: 0;
    }

    .small-header-subtitle {
        font-size: 0.78rem;
        opacity: 0.70;
        margin-top: 0.15rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.0rem;
        padding-left: 0.65rem;
        padding-right: 0.65rem;
        font-size: 0.85rem;
    }

    section[data-testid="stSidebar"] {
        min-width: 335px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Force a clean light-mode appearance regardless of browser/system theme.
st.markdown(
    """
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #f7f9fc !important;
        color: #111827 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(247,249,252,0.92) !important;
    }
    section[data-testid="stSidebar"] {
        background: #ffffff !important;
        border-right: 1px solid #e5e7eb !important;
    }
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    .small-header {
        background: rgba(255,255,255,0.96) !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.65rem !important;
        color: #111827 !important;
        box-shadow: 0 1px 8px rgba(15,23,42,0.06) !important;
    }
    .small-header-title, .small-header-subtitle {
        color: #111827 !important;
    }
    div[data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.65rem !important;
    }
    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 0.55rem !important;
        padding: 0.35rem 0.55rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff !important;
        border-bottom: 1px solid #e5e7eb !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #111827 !important;
    }
    .note-box-light {
        border-left: 3px solid #2563eb;
        padding: 0.45rem 0.6rem;
        background: #eff6ff;
        border-radius: 0.35rem;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)



# Strong high-contrast light-mode override for Streamlit/BaseWeb widgets.
st.markdown(
    """
    <style>
    :root { color-scheme: light !important; }

    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
        background: #ffffff !important;
        color: #111827 !important;
    }

    [data-testid="stHeader"], [data-testid="stToolbar"] {
        background: rgba(255,255,255,0.96) !important;
        color: #111827 !important;
    }

    section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background: #ffffff !important;
        color: #111827 !important;
        border-right: 1px solid #d1d5db !important;
    }

    section[data-testid="stSidebar"] *:not(svg):not(path) {
        color: #111827 !important;
    }

    .small-header {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.65rem !important;
        color: #111827 !important;
        box-shadow: 0 1px 8px rgba(15,23,42,0.08) !important;
    }

    .small-header-title,
    .small-header-subtitle,
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #111827 !important;
    }

    div[data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 0.65rem !important;
        box-shadow: 0 1px 3px rgba(15,23,42,0.05) !important;
        overflow: hidden !important;
    }

    div[data-testid="stExpander"] details summary {
        background: #f8fafc !important;
        color: #111827 !important;
        border-bottom: 1px solid #e5e7eb !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"],
    input,
    textarea {
        background-color: #ffffff !important;
        color: #111827 !important;
        border-color: #64748b !important;
        caret-color: #111827 !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] span,
    div[data-baseweb="select"] div,
    div[data-baseweb="popover"] div,
    [role="option"] {
        color: #111827 !important;
        background-color: #ffffff !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] ul,
    ul[role="listbox"] {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #94a3b8 !important;
    }

    button,
    [data-testid="stBaseButton-secondary"],
    [data-testid="stBaseButton-primary"] {
        background: #ffffff !important;
        color: #111827 !important;
        border: 1px solid #64748b !important;
        box-shadow: none !important;
    }

    [data-testid="stBaseButton-primary"] {
        background: #2563eb !important;
        color: #ffffff !important;
        border-color: #1d4ed8 !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] div {
        background: #ffffff !important;
        color: #111827 !important;
        border-color: #94a3b8 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: #ffffff !important;
        border-bottom: 1px solid #d1d5db !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #111827 !important;
        background: #ffffff !important;
    }

    .stTabs [aria-selected="true"] {
        color: #dc2626 !important;
        border-bottom-color: #dc2626 !important;
    }

    div[data-testid="stMetric"] {
        background: #ffffff !important;
        border: 1px solid #d1d5db !important;
        color: #111827 !important;
    }

    .note-box-light {
        border-left: 3px solid #2563eb;
        padding: 0.45rem 0.6rem;
        background: #eff6ff;
        border-radius: 0.35rem;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        color: #1f2937;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="small-header">
        <div class="small-header-title">Fiber Photometry Analysis</div>
        <div class="small-header-subtitle">
            Upload .ppd or .csv files → run Jones-style processing → inspect graphs → visually zoom/crop without changing the CSV.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# JONES-STYLE PROCESSING SETTINGS FOR CSV INPUTS
# =============================================================================

LOWPASS_HZ = 1.0
FILTER_ORDER = 3
REGRESSION_WINDOW_SEC = 60.0
BASELINE_PERCENTILE = 10.0
EXPORT_HZ_DEFAULT = 1.0

PLOTLY_CONFIG = {
    "scrollZoom": False,  # prevents mouse/trackpad scroll from zooming graphs
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

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
    """Return standard x-axis ticks every 30 minutes, labeled HH:MM:SS."""
    interval = 30 * 60  # 30 minutes in seconds
    min_sec = float(min_sec) if np.isfinite(min_sec) else 0.0
    max_sec = float(max_sec) if np.isfinite(max_sec) else max(min_sec + interval, interval)
    if max_sec <= min_sec:
        max_sec = min_sec + interval

    start_tick = max(0.0, np.floor(min_sec / interval) * interval)
    end_tick = np.ceil(max_sec / interval) * interval
    ticks = np.arange(start_tick, end_tick + interval * 0.5, interval, dtype=float)
    if ticks.size == 0:
        ticks = np.array([0.0, float(interval)])
    return ticks.tolist(), [seconds_to_hms(t) for t in ticks]


def apply_hms_xaxis(fig, df, rows=None):
    """Show standard 30-minute x-axis tick marks as HH:MM:SS without overlapping subplot text."""
    if "time_sec" not in df.columns or df.empty:
        return fig
    min_sec = float(np.nanmin(df["time_sec"]))
    max_sec = float(np.nanmax(df["time_sec"]))
    tickvals, ticktext = make_time_ticks(min_sec, max_sec)

    if rows is None:
        fig.update_xaxes(
            title_text="Time (HH:MM:SS)",
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
                title_text="Time (HH:MM:SS)" if show_labels else "",
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

    # If time appears to be milliseconds, convert to seconds.
    if df["time_sec"].max() > 100000 and df["time_sec"].median() > 1000:
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


def process_raw_csv_dataframe(df, roi):
    df = standardize_time(df)
    uv_col, sig_col = find_raw_channel_columns(df, roi)
    if uv_col is None or sig_col is None:
        raise ValueError(
            "Could not identify raw 405/isobestic and 465/signal columns. "
            "Use columns like time_sec,BLA_iso,BLA_sig or time_sec,BLA_uv_raw,BLA_sig_raw."
        )

    uv_raw = pd.to_numeric(df[uv_col], errors="coerce").to_numpy(dtype=float)
    sig_raw = pd.to_numeric(df[sig_col], errors="coerce").to_numpy(dtype=float)

    good = np.asarray(np.isfinite(df["time_sec"]) & np.isfinite(uv_raw) & np.isfinite(sig_raw))
    df = df.loc[good].reset_index(drop=True)
    uv_raw = uv_raw[good]
    sig_raw = sig_raw[good]

    fs = estimate_fs(df["time_sec"].to_numpy())
    uv_filt = lowpass(uv_raw, fs)
    sig_filt = lowpass(sig_raw, fs)
    uv_fit, delta_f = rolling_filtered_to_raw_fit(uv_raw, sig_raw, uv_filt, sig_filt, fs)

    f0 = float(np.nanpercentile(uv_raw, BASELINE_PERCENTILE))
    if not np.isfinite(f0) or f0 == 0:
        raise ValueError(f"Invalid F0 from CSV: {f0}")

    dff = 100.0 * delta_f / f0
    sd = np.nanstd(dff, ddof=1)
    z_dff = (dff - np.nanmean(dff)) / sd if sd and np.isfinite(sd) else np.full_like(dff, np.nan)

    out = pd.DataFrame({
        "time_sec": df["time_sec"].to_numpy(),
        "elapsed_hours": df["time_sec"].to_numpy() / 3600.0,
        f"{roi}_uv_raw": uv_raw,
        f"{roi}_sig_raw": sig_raw,
        f"{roi}_uv_filt_1Hz_lowpass": uv_filt,
        f"{roi}_sig_filt_1Hz_lowpass": sig_filt,
        f"{roi}_uv_fit": uv_fit,
        f"{roi}_deltaF": delta_f,
        f"{roi}_dFF": dff,
        f"{roi}_z_dFF": z_dff,
    })
    out["elapsed_hhmmss"] = [seconds_to_hms(x) for x in out["time_sec"]]

    settings = {
        "input_type": "csv_raw",
        "uv_column_used": uv_col,
        "sig_column_used": sig_col,
        "sampling_rate_estimated_hz": fs,
        "lowpass_hz": LOWPASS_HZ,
        "filter_order": FILTER_ORDER,
        "regression_window_sec": REGRESSION_WINDOW_SEC,
        "baseline_method": "uv_raw_percentile_session",
        "baseline_percentile": BASELINE_PERCENTILE,
        "F0": f0,
    }
    return out, settings


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


def finish_fig(fig, height, show_legend=True):
    fig.update_layout(
        height=height,
        margin=dict(l=45, r=25, t=70, b=40),
        hovermode="x unified",
        dragmode="pan",  # drag pans by default; scroll-wheel zoom is disabled in PLOTLY_CONFIG
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        showlegend=show_legend,
    )
    return fig


def make_overview_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=1050):
    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=(
            "Raw 465 and 405 together",
            "Correction check: 465 signal and fitted 405",
            "Delta-F corrected signal",
            "dFF (%)",
            "z-scored dFF",
        ),
    )
    x = df["time_sec"]
    if f"{roi}_sig_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465", showlegend=False), row=2, col=1)
    if f"{roi}_uv_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="UV Raw / 405"), row=1, col=1)
    if f"{roi}_uv_fit" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_fit"], mode="lines", name="UV Fit / fitted 405"), row=2, col=1)
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F"), row=3, col=1)
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)"), row=4, col=1)
    if f"{roi}_z_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_z_dFF"], mode="lines", name="z-dFF"), row=5, col=1)

    fig.update_yaxes(title_text="Raw", row=1, col=1)
    fig.update_yaxes(title_text="Fit", row=2, col=1)
    fig.update_yaxes(title_text="Delta-F", row=3, col=1)
    fig.update_yaxes(title_text="dFF (%)", row=4, col=1)
    fig.update_yaxes(title_text="z-dFF", row=5, col=1)
    apply_hms_xaxis(fig, df, rows=[1, 2, 3, 4, 5])

    if limits:
        apply_y_range(fig, 1, limits.get("raw"))
        apply_y_range(fig, 2, limits.get("fit"))
        apply_y_range(fig, 3, limits.get("deltaF"))
        apply_y_range(fig, 4, limits.get("dFF"))
        apply_y_range(fig, 5, limits.get("z_dFF"))

    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1, 2, 3, 4, 5], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1, 2, 3, 4, 5], show_labels=show_event_labels)
    return finish_fig(fig, height=height)


def make_raw_independent_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=650):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("465 nm calcium-dependent signal", "405 nm isosbestic/control signal"))
    x = df["time_sec"]
    if f"{roi}_sig_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="465 raw"), row=1, col=1)
    if f"{roi}_uv_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="405 raw"), row=2, col=1)
    fig.update_yaxes(title_text="465 raw", row=1, col=1)
    fig.update_yaxes(title_text="405 raw", row=2, col=1)
    apply_hms_xaxis(fig, df, rows=[1, 2])
    if limits:
        apply_y_range(fig, 1, limits.get("raw"))
        apply_y_range(fig, 2, limits.get("raw"))
    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1, 2], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1, 2], show_labels=show_event_labels)
    return finish_fig(fig, height=height)


def make_processed_figure(df, roi, limits=None, events=None, exclusions=None, show_exclusions=True, show_event_labels=True, height=760):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=("Delta-F", "dFF (%) - main normalized trace", "z-scored dFF"))
    x = df["time_sec"]
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F"), row=1, col=1)
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)"), row=2, col=1)
    if f"{roi}_z_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_z_dFF"], mode="lines", name="z-dFF"), row=3, col=1)
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
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F"), secondary_y=False)
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)"), secondary_y=True)
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
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465", showlegend=False), row=2, col=1)
    if f"{roi}_uv_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="UV Raw / 405"), row=1, col=1)
    if f"{roi}_uv_fit" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_fit"], mode="lines", name="UV Fit / fitted 405"), row=2, col=1)
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F"), row=3, col=1)

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
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="465 raw"), row=1, col=1)
    if f"{roi}_uv_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="405 raw"), row=2, col=1)
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
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)"), row=1, col=1)
    fig.update_yaxes(title_text="dFF (%)", row=1, col=1)
    apply_hms_xaxis(fig, df, rows=[1])
    if limits:
        apply_y_range(fig, 1, limits.get("dFF"))
    add_visual_exclusion_shapes(fig, exclusions or [], rows=[1], show=show_exclusions)
    add_events_to_fig(fig, events or [], rows=[1], show_labels=show_event_labels)
    return finish_fig(fig, height=height, df=df)

# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

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
            ["Overview", "Raw channels", "Main dFF"],
            index=0,
            disabled=not focus_mode,
        )
        show_event_labels = st.checkbox("Show event labels above graphs", value=True)
        show_hidden_region_shading = st.checkbox("Mark hidden regions with pale gray shading", value=True)
        st.caption("Mouse/trackpad scroll zoom is disabled. Use graph size/focus mode to view graphs larger.")

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
            with st.expander(f"Hidden region {i + 1} timing", expanded=True):
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
            with st.expander(f"Event / epoch {i + 1}: name and timing", expanded=True):
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

            with st.expander(f"Event / epoch {i + 1}: color and shading", expanded=False):
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
                                all_results.append({
                                    "name": csv_path.parent.name if csv_path.parent.name else csv_path.stem,
                                    "df": df,
                                    "settings": {"input_type": "ppd", "source_csv": str(csv_path)},
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
            if kind == "Overview":
                fig = make_overview_figure(
                    view_df, roi_name, manual_limits,
                    events=event_annotations,
                    exclusions=visual_exclusions,
                    show_exclusions=show_hidden_region_shading,
                    show_event_labels=show_event_labels,
                    height=overview_h,
                )
            elif kind == "Raw channels":
                fig = make_raw_independent_figure(
                    view_df, roi_name, manual_limits,
                    events=event_annotations,
                    exclusions=visual_exclusions,
                    show_exclusions=show_hidden_region_shading,
                    show_event_labels=show_event_labels,
                    height=raw_h,
                )
            else:
                fig = make_main_dff_figure(
                    view_df, roi_name, manual_limits,
                    events=event_annotations,
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

        tabs = st.tabs(["Overview", "Raw channels", "Main dFF", "Data preview", "Settings"])

        with tabs[0]:
            render_chart("Overview")
        with tabs[1]:
            render_chart("Raw channels")
        with tabs[2]:
            render_chart("Main dFF")
        with tabs[3]:
            st.caption("Preview of the displayed data window. The downloadable CSV remains the full processed data.")
            st.dataframe(view_df.head(500), use_container_width=True, height=360)
            st.download_button(
                label=f"Download full processed CSV for {result['name']}",
                data=full_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_name(result['name'])}_processed_output.csv",
                mime="text/csv",
            )
        with tabs[4]:
            st.json(result.get("settings", {}))

    st.divider()
    st.download_button(
        label="Download all full processed outputs as ZIP",
        data=make_zip_from_results(results, roi_name),
        file_name="photometry_analysis_outputs.zip",
        mime="application/zip",
        use_container_width=True,
    )
