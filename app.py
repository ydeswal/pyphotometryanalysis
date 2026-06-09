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

# =============================================================================
# HELPERS
# =============================================================================

def seconds_to_hms(seconds):
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def filter_visual_window(df, start_hr, end_hr):
    if start_hr is None or end_hr is None:
        return df
    return df[(df["elapsed_hours"] >= start_hr) & (df["elapsed_hours"] <= end_hr)].copy()


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


def make_overview_figure(df, roi, limits=None):
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
    x = df["elapsed_hours"]
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
    fig.update_xaxes(title_text="Time (hours)", row=5, col=1)

    if limits:
        apply_y_range(fig, 1, limits.get("raw"))
        apply_y_range(fig, 2, limits.get("fit"))
        apply_y_range(fig, 3, limits.get("deltaF"))
        apply_y_range(fig, 4, limits.get("dFF"))
        apply_y_range(fig, 5, limits.get("z_dFF"))

    fig.update_layout(
        height=1050,
        margin=dict(l=45, r=25, t=60, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    return fig


def make_raw_independent_figure(df, roi, limits=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=("465 nm calcium-dependent signal", "405 nm isosbestic/control signal"))
    x = df["elapsed_hours"]
    if f"{roi}_sig_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="465 raw"), row=1, col=1)
    if f"{roi}_uv_raw" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="405 raw"), row=2, col=1)
    fig.update_yaxes(title_text="465 raw", row=1, col=1)
    fig.update_yaxes(title_text="405 raw", row=2, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    if limits:
        apply_y_range(fig, 1, limits.get("raw"))
        apply_y_range(fig, 2, limits.get("raw"))
    fig.update_layout(height=650, margin=dict(l=45, r=25, t=60, b=40), hovermode="x unified")
    return fig


def make_processed_figure(df, roi, limits=None):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.055,
                        subplot_titles=("Delta-F", "dFF (%) - main normalized trace", "z-scored dFF"))
    x = df["elapsed_hours"]
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F"), row=1, col=1)
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)"), row=2, col=1)
    if f"{roi}_z_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_z_dFF"], mode="lines", name="z-dFF"), row=3, col=1)
    fig.update_yaxes(title_text="Delta-F", row=1, col=1)
    fig.update_yaxes(title_text="dFF (%)", row=2, col=1)
    fig.update_yaxes(title_text="z-dFF", row=3, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
    if limits:
        apply_y_range(fig, 1, limits.get("deltaF"))
        apply_y_range(fig, 2, limits.get("dFF"))
        apply_y_range(fig, 3, limits.get("z_dFF"))
    fig.update_layout(height=760, margin=dict(l=45, r=25, t=60, b=40), hovermode="x unified")
    return fig


def make_dual_delta_dff_figure(df, roi, limits=None):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df["elapsed_hours"]
    if f"{roi}_deltaF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F"), secondary_y=False)
    if f"{roi}_dFF" in df.columns:
        fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="dFF (%)"), secondary_y=True)
    fig.update_xaxes(title_text="Time (hours)")
    fig.update_yaxes(title_text="Delta-F", secondary_y=False)
    fig.update_yaxes(title_text="dFF (%)", secondary_y=True)
    if limits:
        if limits.get("deltaF") is not None:
            fig.update_yaxes(range=list(limits["deltaF"]), secondary_y=False)
        if limits.get("dFF") is not None:
            fig.update_yaxes(range=list(limits["dFF"]), secondary_y=True)
    fig.update_layout(height=560, margin=dict(l=45, r=45, t=40, b=40), hovermode="x unified")
    return fig


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
# SIDEBAR CONTROLS
# =============================================================================

with st.sidebar:
    st.header("Inputs")
    uploaded_files = st.file_uploader(
        "Upload .ppd or .csv file(s)",
        type=["ppd", "csv"],
        accept_multiple_files=True,
        help="PPD files are sent to ana.py. CSV files are either processed directly or displayed if already processed.",
    )
    roi_name = st.text_input("ROI name", value="BLA")
    scale_mode = st.selectbox("Auto y-axis scaling", ["full", "robust", "none"], index=0)
    save_full = st.checkbox("Save full 20 Hz processed CSV for PPD runs", value=False)

    st.divider()
    st.header("Visual window only")
    use_window = st.checkbox(
        "Cut/zoom to a time region for viewing",
        value=False,
        help="This only changes what you see on the webpage. It does not change the saved processed CSV.",
    )
    start_hr = st.number_input("Start hour", value=0.0, min_value=0.0, step=0.1)
    end_hr = st.number_input("End hour", value=7.5, min_value=0.0, step=0.1)

    st.divider()
    st.header("Manual y-scales")
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

    st.success(f"Loaded {len(results)} recording(s).")
    c1, c2, c3, c4 = st.columns(4)
    max_duration = max(float(r["df"]["elapsed_hours"].max()) for r in results)
    c1.metric("Recordings loaded", len(results))
    c2.metric("Max duration", f"{max_duration:.2f} hr")
    c3.metric("ROI", roi_name)
    c4.metric("Visual crop", "ON" if use_window else "OFF")

    with st.expander("Current y-axis limits and visual window"):
        st.json({
            "auto_limits": {k: list(v) if v else None for k, v in auto.items()},
            "active_limits_after_manual_override": {k: list(v) if v else None for k, v in manual_limits.items()},
            "visual_window_hours": [start_hr, end_hr] if use_window else "full recording",
            "note": "Visual window affects only webpage display, not downloaded processed CSV files.",
        })

    for idx, result in enumerate(results, start=1):
        full_df = result["df"]
        view_df = filter_visual_window(full_df, start_hr, end_hr) if use_window else full_df
        if view_df.empty:
            st.warning(f"{result['name']}: visual window has no data.")
            continue

        st.markdown(f"## Recording {idx}: `{result['name']}`")
        tabs = st.tabs(["Overview", "Raw channels", "Processed", "Delta-F + dFF", "Data preview", "Settings"])

        with tabs[0]:
            st.plotly_chart(make_overview_figure(view_df, roi_name, manual_limits), use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})
        with tabs[1]:
            st.plotly_chart(make_raw_independent_figure(view_df, roi_name, manual_limits), use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})
        with tabs[2]:
            st.plotly_chart(make_processed_figure(view_df, roi_name, manual_limits), use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})
        with tabs[3]:
            st.plotly_chart(make_dual_delta_dff_figure(view_df, roi_name, manual_limits), use_container_width=True,
                            config={"scrollZoom": True, "displaylogo": False})
        with tabs[4]:
            st.caption("Preview of the displayed data window. The downloadable CSV remains the full processed data.")
            st.dataframe(view_df.head(500), use_container_width=True, height=360)
            st.download_button(
                label=f"Download full processed CSV for {result['name']}",
                data=full_df.to_csv(index=False).encode("utf-8"),
                file_name=f"{safe_name(result['name'])}_processed_output.csv",
                mime="text/csv",
            )
        with tabs[5]:
            st.json(result.get("settings", {}))

    st.divider()
    st.download_button(
        label="Download all full processed outputs as ZIP",
        data=make_zip_from_results(results, roi_name),
        file_name="photometry_analysis_outputs.zip",
        mime="application/zip",
        use_container_width=True,
    )
