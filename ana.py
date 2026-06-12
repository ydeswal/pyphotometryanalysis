#!/usr/bin/env python3
"""
Lowell/Douglass-style pyPhotometry .ppd analysis WITHOUT event timestamps,
with MATCHED AXIS SCALES across multiple recordings.

Why this version exists
-----------------------
When you make one interactive graph per recording, Plotly/Matplotlib normally
auto-scale each graph separately. That makes two recordings look different even
when the signal magnitude is similar. This version processes all input .ppd files
first, calculates one shared set of x/y limits, and then applies those same limits
to every output plot and interactive HTML file.

Install once:
    pip install numpy pandas scipy matplotlib plotly

Run one file:
    python ana_same_scales.py recording1.ppd --roi BLA

Run two or more files with the same scales:
    python ana_same_scales.py recording1.ppd recording2.ppd --roi BLA

Recommended for direct comparison:
    python ana_same_scales.py rec1.ppd rec2.ppd --roi BLA --scale-mode full

If one recording has a huge artifact spike and you want the useful signal range:
    python ana_same_scales.py rec1.ppd rec2.ppd --roi BLA --scale-mode robust

Outputs
-------
A parent output folder with one subfolder per recording. Every subfolder contains:
  Plot A - raw traces together
  Plot B - raw traces independently across the continuous recording
  Plot C - stacked corrected chunks
  Plot D - correction impact
  Plot E - raw 465 and 405 on the same graph
  Plot F - raw 465 and 405 independently
  Plot G - Delta-F, dFF, and z-dFF
  Plot H - Delta-F and dFF on the same graph
  Interactive HTML - zoomable browser graph
  processed CSV + settings files

Important
---------
This script does not require or use event/timestamp data.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import shutil
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, medfilt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# USER SETTINGS YOU CAN EDIT
# =============================================================================

DEFAULT_PPD_PATHS = [
    Path("/Users/yashdeswal/Downloads/bluetail_BLA_6_4_26-2026-06-04-111310.ppd"),
]
DEFAULT_ROI_NAME = "BLA"

# Your pyPhotometry / Doric 2EX_1EM_pulsed stream layout.
# Change this only if your acquisition mode has a different stream order.
COLUMN_MAP = {
    "465_on": 0,
    "465_dark": 1,
    "405_on": 2,
    "405_dark": 3,
}

# Lowell/Douglass-style defaults used here.
LOWPASS_HZ = 10.0
FILTER_ORDER = 3
REGRESSION_WINDOW_SEC = 60.0
BASELINE_PERCENTILE = 10.0  # retained only for backward compatibility; Lowell F0 does not use this

# Lowell/Douglass paper method settings.
MEDIAN_FILTER_SEC = 1.0          # paper says median-filtered; kernel was not specified in the PDF
F0_LOWPASS_HZ = 0.001           # F0 = 0.001 Hz low-pass filtered 465 nm signal
LONG_TERM_SMOOTH_SEC = 30 * 60  # 30-minute running average for long-term traces

# Plot/export settings.
EXPORT_HZ = 1.0              # graphs and main CSV are downsampled to 1 Hz
STACKED_CHUNK_SEC = 600.0    # Plot C chunk length in seconds

# Padding added around y-axis limits. Example: 0.05 = 5% padding.
AXIS_PADDING_FRACTION = 0.05

# Robust scale percentiles used only with --scale-mode robust.
ROBUST_LOW_PERCENTILE = 0.5
ROBUST_HIGH_PERCENTILE = 99.5


# =============================================================================
# BASIC HELPERS
# =============================================================================

def seconds_to_hms(seconds: float) -> str:
    seconds = int(round(float(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def safe_stem(path: Path) -> str:
    """Make a filesystem-safe output folder name."""
    stem = path.stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return stem[:120] if len(stem) > 120 else stem


def find_ppd_file(path: Path) -> Path:
    if path.exists():
        return path
    ppds = sorted(Path.cwd().glob("*.ppd"))
    if len(ppds) == 1:
        print(f"Using the only .ppd file found in this folder: {ppds[0]}")
        return ppds[0]
    raise FileNotFoundError(
        f"Could not find: {path}\n"
        "Either edit DEFAULT_PPD_PATHS in the script or run:\n"
        "    python ana_same_scales.py /path/to/file1.ppd /path/to/file2.ppd"
    )


def read_ppd(ppd_path: Path):
    """
    Read pyPhotometry .ppd file.

    .ppd structure:
      first 2 bytes       = JSON header size
      next header bytes   = JSON header
      remaining bytes     = uint16 interleaved data

    pyPhotometry stores analog value in upper bits, so the analog conversion uses
    raw >> 1. The lowest bit is digital state.
    """
    with open(ppd_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        header = json.loads(f.read(header_size).decode("utf-8"))
        raw = np.frombuffer(f.read(), dtype="<u2")

    n_streams = int(header["n_analog_signals"]) + int(header["n_digital_signals"])
    if raw.size % n_streams != 0:
        raise ValueError("Raw .ppd data length is not divisible by number of streams.")

    data = raw.reshape(-1, n_streams)

    volts_per_division = header.get("volts_per_division", [1.0])
    if len(volts_per_division) == 1:
        vpd = [float(volts_per_division[0])] * n_streams
    else:
        vpd = [float(x) for x in volts_per_division]

    volts = np.zeros_like(data, dtype=float)
    for i in range(n_streams):
        volts[:, i] = (data[:, i] >> 1).astype(float) * vpd[min(i, len(vpd) - 1)]

    fs = float(header["sampling_rate"])
    return header, fs, volts


def extract_channels(volts: np.ndarray):
    """
    Extract dark-subtracted 465 and 405 signals.

    Formula:
      sig_raw = 465_on - 465_dark
      uv_raw  = 405_on - 405_dark

    Simple meaning:
      subtract the LED-off/background level from each LED-on measurement.
    """
    required_col = max(COLUMN_MAP.values())
    if volts.shape[1] <= required_col:
        raise ValueError(
            f"This file has {volts.shape[1]} streams, but COLUMN_MAP requires column {required_col}."
        )
    sig_raw = volts[:, COLUMN_MAP["465_on"]] - volts[:, COLUMN_MAP["465_dark"]]
    uv_raw = volts[:, COLUMN_MAP["405_on"]] - volts[:, COLUMN_MAP["405_dark"]]
    return uv_raw, sig_raw


# =============================================================================
# LOWELL / DOUGLASS PAPER-STYLE PHOTOMETRY MATH
# =============================================================================

def _odd_kernel_samples(window_sec: float, fs: float, minimum: int = 3) -> int:
    """Return an odd integer sample window for median filtering."""
    k = int(round(float(window_sec) * float(fs)))
    k = max(k, minimum)
    if k % 2 == 0:
        k += 1
    return k


def median_filter_trace(values: np.ndarray, fs: float):
    """
    Median-filter the trace.

    The Lowell/Douglass paper states that all data were median-filtered before
    the 10 Hz low-pass filter. The PDF does not specify the median-filter kernel,
    so this script exposes MEDIAN_FILTER_SEC as an editable setting.
    """
    if MEDIAN_FILTER_SEC <= 0:
        return values.copy()
    kernel = _odd_kernel_samples(MEDIAN_FILTER_SEC, fs)
    # scipy.signal.medfilt pads edges with zeros, so replace only if finite data exist.
    return medfilt(values, kernel_size=kernel)


def lowpass(values: np.ndarray, fs: float, cutoff_hz: float | None = None, order: int | None = None):
    """
    Zero-phase Butterworth low-pass filter.

    Lowell/Douglass method:
      - photometry traces: 10 Hz cutoff
      - F0 baseline: 0.001 Hz cutoff on the 465 nm signal

    If cutoff equals/exceeds Nyquist, the function automatically lowers it just
    below Nyquist so scipy can run safely.
    """
    if cutoff_hz is None:
        cutoff_hz = LOWPASS_HZ
    if order is None:
        order = FILTER_ORDER

    nyquist = fs / 2
    if cutoff_hz <= 0:
        return values.copy()
    safe_cutoff = min(float(cutoff_hz), nyquist * 0.99)
    if safe_cutoff <= 0:
        return values.copy()

    sos = butter(order, safe_cutoff, btype="lowpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, values)
    except ValueError:
        # Very short recordings may fail filtfilt pad-length requirements.
        # Fall back to the original trace rather than crashing.
        warnings.warn(
            f"Could not apply {cutoff_hz:g} Hz low-pass filter, likely because the recording is too short. "
            "Returning the unfiltered trace for this step."
        )
        return values.copy()


def global_linear_405_fit_to_465(uv_signal, sig_signal):
    """
    Lowell/Douglass motion correction:
      find a single linear best fit to the 405 nm signal, then subtract the fitted
      405 signal from the 465 nm signal.

    Formula:
      fitted_405(t) = slope * signal_405(t) + intercept
      DeltaF(t)    = signal_465(t) - fitted_405(t)
    """
    good = np.isfinite(uv_signal) & np.isfinite(sig_signal)
    if good.sum() < 3:
        raise ValueError("Not enough finite samples for 405-to-465 linear fit.")

    slope, intercept = np.polyfit(uv_signal[good], sig_signal[good], 1)
    uv_fit = slope * uv_signal + intercept
    delta_f = sig_signal - uv_fit
    return uv_fit, delta_f, slope, intercept


def rolling_mean_trace(values: np.ndarray, fs: float, window_sec: float, center: bool = True):
    """Running average used for long-term traces such as the 30 min mean in Figure 1N/2A."""
    window = max(int(round(float(window_sec) * float(fs))), 1)
    return (
        pd.Series(values)
        .rolling(window=window, center=center, min_periods=max(1, window // 5))
        .mean()
        .bfill()
        .ffill()
        .to_numpy()
    )


def zscore(values: np.ndarray):
    """Session z-score."""
    mean = np.nanmean(values)
    sd = np.nanstd(values, ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.full_like(values, np.nan, dtype=float)
    return (values - mean) / sd


def process_photometry(uv_raw, sig_raw, fs):
    """
    Run Lowell/Douglass-style processing.

    Paper method implemented here:
      1. Median-filter both 405 and 465 traces.
      2. Low-pass filter both traces at 10 Hz.
      3. Motion correction: linear best fit to 405, subtract fitted 405 from 465.
      4. F0: 0.001 Hz low-pass filtered 465 nm trace across the whole session.
      5. DF/F0 = (465 - fitted 405) / F0.
      6. z_DF/F0 = session z-score of DF/F0.
      7. 30 min running-average columns are also generated for long-term plots.
    """
    # 1–2. Median + 10 Hz low-pass.
    uv_med = median_filter_trace(uv_raw, fs)
    sig_med = median_filter_trace(sig_raw, fs)

    uv_filt = lowpass(uv_med, fs, cutoff_hz=LOWPASS_HZ)
    sig_filt = lowpass(sig_med, fs, cutoff_hz=LOWPASS_HZ)

    # 3. Global linear 405 fit, not rolling regression.
    uv_fit, delta_f, slope, intercept = global_linear_405_fit_to_465(uv_filt, sig_filt)

    # 4. F0 is the 0.001 Hz low-pass filtered 465 trace.
    f0_trace = lowpass(sig_filt, fs, cutoff_hz=F0_LOWPASS_HZ)

    # Protect against divide-by-zero or negative/invalid baseline values.
    eps = np.nanmedian(np.abs(f0_trace)) * 1e-12
    if not np.isfinite(eps) or eps == 0:
        eps = 1e-12
    f0_safe = np.where(np.isfinite(f0_trace) & (np.abs(f0_trace) > eps), f0_trace, np.nan)

    # 5. Lowell DF/F0 ratio. This is intentionally NOT multiplied by 100.
    dff = delta_f / f0_safe

    # Optional percent copy for people who want percent units.
    dff_percent = 100.0 * dff

    # 6. Session z-score of the Lowell DF/F0 ratio.
    z_dff = zscore(dff)

    # 7. 30-minute running means for paper-style long-term viewing.
    dff_30min = rolling_mean_trace(dff, fs, LONG_TERM_SMOOTH_SEC, center=True)
    z_dff_30min = rolling_mean_trace(z_dff, fs, LONG_TERM_SMOOTH_SEC, center=True)

    return {
        "uv_med": uv_med,
        "sig_med": sig_med,
        "uv_filt": uv_filt,
        "sig_filt": sig_filt,
        "uv_fit": uv_fit,
        "deltaF": delta_f,
        "F0_trace": f0_trace,
        "F0": float(np.nanmedian(f0_trace)),
        "dFF": dff,
        "dFF_percent": dff_percent,
        "z_dFF": z_dff,
        "dFF_30min_mean": dff_30min,
        "z_dFF_30min_mean": z_dff_30min,
        "slope": slope,
        "intercept": intercept,
    }


def make_dataframe(time_sec, uv_raw, sig_raw, processed, roi):
    df = pd.DataFrame({
        "time_sec": time_sec,
        "elapsed_hours": time_sec / 3600,
        f"{roi}_uv_raw": uv_raw,
        f"{roi}_sig_raw": sig_raw,

        # Median-filtered and 10 Hz low-pass-filtered traces.
        f"{roi}_uv_median_filtered": processed["uv_med"],
        f"{roi}_sig_median_filtered": processed["sig_med"],
        f"{roi}_uv_filt_10Hz_lowpass": processed["uv_filt"],
        f"{roi}_sig_filt_10Hz_lowpass": processed["sig_filt"],

        # Motion correction and Lowell/Douglass DF/F0.
        f"{roi}_uv_fit": processed["uv_fit"],
        f"{roi}_deltaF": processed["deltaF"],
        f"{roi}_F0_0p001Hz": processed["F0_trace"],

        # Main app-compatible column: Lowell DF/F0 ratio, not percent.
        f"{roi}_dFF": processed["dFF"],
        f"{roi}_DF_F0": processed["dFF"],

        # Optional percent copy.
        f"{roi}_dFF_percent": processed["dFF_percent"],

        # Z-score and paper-style 30 min running averages.
        f"{roi}_z_dFF": processed["z_dFF"],
        f"{roi}_dFF_30min_mean": processed["dFF_30min_mean"],
        f"{roi}_z_dFF_30min_mean": processed["z_dFF_30min_mean"],
    })
    df["elapsed_hhmmss"] = [seconds_to_hms(x) for x in df["time_sec"]]
    return df


def downsample(df, fs, export_hz):
    if export_hz >= fs:
        return df.copy()
    bin_size = max(int(round(fs / export_hz)), 1)
    tmp = df.copy()
    tmp["export_bin"] = np.arange(len(tmp)) // bin_size
    numeric_cols = tmp.select_dtypes(include=[np.number]).columns
    out = tmp.groupby("export_bin", as_index=False)[numeric_cols].mean()
    out["elapsed_hhmmss"] = [seconds_to_hms(x) for x in out["time_sec"]]
    return out.drop(columns=["export_bin"], errors="ignore")


# =============================================================================
# GLOBAL AXIS LIMITS
# =============================================================================

def _finite_values(values):
    arr = np.asarray(values, dtype=float).ravel()
    return arr[np.isfinite(arr)]


def padded_limits(values, mode="full", pad_fraction=AXIS_PADDING_FRACTION):
    """Return padded y-limits from a list/array of values."""
    arr = _finite_values(values)
    if arr.size == 0:
        return [0.0, 1.0]

    if mode == "robust":
        lo = float(np.nanpercentile(arr, ROBUST_LOW_PERCENTILE))
        hi = float(np.nanpercentile(arr, ROBUST_HIGH_PERCENTILE))
    else:
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return [0.0, 1.0]
    if lo == hi:
        pad = abs(lo) * 0.05 if lo != 0 else 1.0
        return [lo - pad, hi + pad]

    span = hi - lo
    pad = span * pad_fraction
    return [lo - pad, hi + pad]


def compute_axis_limits(recordings, roi: str, scale_mode: str):
    """
    Calculate one common axis scale dictionary across all recordings.

    These limits are then applied to every static and interactive output, so the
    same plot from different recordings has the same y-scale.
    """
    if scale_mode == "none":
        return None

    dfs = [r["df_plot"] for r in recordings]
    max_hours = max(float(df["elapsed_hours"].max()) for df in dfs)
    max_seconds = max(float(df["time_sec"].max()) for df in dfs)

    def concat(cols):
        pieces = []
        for df in dfs:
            for c in cols:
                if c in df.columns:
                    pieces.append(df[c].to_numpy())
        return np.concatenate(pieces) if pieces else np.array([])

    uv_raw = f"{roi}_uv_raw"
    sig_raw = f"{roi}_sig_raw"
    uv_fit = f"{roi}_uv_fit"
    deltaF = f"{roi}_deltaF"
    dFF = f"{roi}_dFF"
    z_dFF = f"{roi}_z_dFF"

    limits = {
        "scale_mode": scale_mode,
        "x_seconds": [0.0, max_seconds],
        "x_hours": [0.0, max_hours],
        "x_chunk_seconds": [0.0, STACKED_CHUNK_SEC],
        "raw_both": padded_limits(concat([uv_raw, sig_raw]), mode=scale_mode),
        "uv_raw": padded_limits(concat([uv_raw]), mode=scale_mode),
        "sig_raw": padded_limits(concat([sig_raw]), mode=scale_mode),
        "fit_check": padded_limits(concat([sig_raw, uv_fit]), mode=scale_mode),
        "deltaF": padded_limits(concat([deltaF]), mode=scale_mode),
        "dFF": padded_limits(concat([dFF]), mode=scale_mode),
        "z_dFF": padded_limits(concat([z_dFF]), mode=scale_mode),
    }

    # Plot C offset step is also shared so stacked plots are directly comparable.
    delta_range = limits["deltaF"][1] - limits["deltaF"][0]
    limits["stacked_step"] = float(delta_range * 1.05 if delta_range > 0 else 1.0)
    return limits


def set_ylim(ax, limits, key):
    if limits is not None and key in limits:
        ax.set_ylim(limits[key])


def set_xlim_hours(ax, limits):
    if limits is not None:
        ax.set_xlim(limits["x_hours"])


def set_xlim_seconds(ax, limits):
    if limits is not None:
        ax.set_xlim(limits["x_seconds"])


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_A(df, outdir, roi, limits):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["time_sec"], df[f"{roi}_uv_raw"], linewidth=0.7, label="UV Raw / 405 isosbestic")
    ax.plot(df["time_sec"], df[f"{roi}_sig_raw"], linewidth=0.7, label="Sig Raw / 465 calcium")
    ax.set_title(f"Plot A: Raw traces - {roi}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Signal (volts / a.u.)")
    set_xlim_seconds(ax, limits)
    set_ylim(ax, limits, "raw_both")
    ax.legend(loc="best")
    return savefig(fig, outdir / f"plot_A_raw_traces_{roi}.png")


def plot_B(df, outdir, roi, limits):
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    axes[0].plot(df["elapsed_hours"], df[f"{roi}_uv_raw"], linewidth=0.7)
    axes[0].set_title("405 nm / UV isosbestic raw")
    axes[0].set_ylabel("405 raw")
    set_ylim(axes[0], limits, "uv_raw")

    axes[1].plot(df["elapsed_hours"], df[f"{roi}_sig_raw"], linewidth=0.7)
    axes[1].set_title("465 nm / signal raw")
    axes[1].set_ylabel("465 raw")
    axes[1].set_xlabel("Time (hours)")
    set_ylim(axes[1], limits, "sig_raw")
    set_xlim_hours(axes[1], limits)

    fig.suptitle(f"Plot B: Continuous raw channels - {roi}")
    return savefig(fig, outdir / f"plot_B_continuous_{roi}.png")


def plot_C(df, outdir, roi, limits):
    fig, ax = plt.subplots(figsize=(16, 9))
    chunk_id = np.floor(df["time_sec"] / STACKED_CHUNK_SEC).astype(int)
    step = limits.get("stacked_step", 1.0) if limits is not None else None
    if step is None:
        y_all = df[f"{roi}_deltaF"].to_numpy()
        robust_range = np.nanpercentile(y_all, 95) - np.nanpercentile(y_all, 5)
        step = robust_range * 1.25 if robust_range > 0 else 1.0

    for i, cid in enumerate(sorted(chunk_id.unique())):
        chunk = df[chunk_id == cid]
        x = chunk["time_sec"] - chunk["time_sec"].iloc[0]
        y = chunk[f"{roi}_deltaF"] + i * step
        ax.plot(x, y, linewidth=0.7)
    ax.set_title(f"Plot C: Stacked Delta-F chunks - {roi}")
    ax.set_xlabel(f"Time within chunk (s); chunk length = {STACKED_CHUNK_SEC:g} s")
    ax.set_ylabel("Stacked Delta-F chunks, vertically offset")
    ax.set_yticks([])
    if limits is not None:
        ax.set_xlim(limits["x_chunk_seconds"])
    return savefig(fig, outdir / f"plot_C_stacked_{roi}.png")


def plot_D(df, outdir, roi, limits):
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(df["elapsed_hours"], df[f"{roi}_sig_raw"], linewidth=0.7, label="Sig Raw / 465")
    axes[0].plot(df["elapsed_hours"], df[f"{roi}_uv_raw"], linewidth=0.7, label="UV Raw / 405")
    axes[0].set_ylabel("Raw signal")
    axes[0].set_title("Raw channels")
    set_ylim(axes[0], limits, "raw_both")
    axes[0].legend(loc="best")

    axes[1].plot(df["elapsed_hours"], df[f"{roi}_sig_raw"], linewidth=0.7, label="Sig Raw / 465")
    axes[1].plot(df["elapsed_hours"], df[f"{roi}_uv_fit"], linewidth=0.7, linestyle="--", label="UV Fit")
    axes[1].set_ylabel("Model fit")
    axes[1].set_title("Fitted 405 artifact estimate")
    set_ylim(axes[1], limits, "fit_check")
    axes[1].legend(loc="best")

    axes[2].plot(df["elapsed_hours"], df[f"{roi}_deltaF"], linewidth=0.7, label="Delta-F corrected")
    axes[2].set_ylabel("Delta-F")
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_title("Corrected signal")
    set_ylim(axes[2], limits, "deltaF")
    set_xlim_hours(axes[2], limits)
    axes[2].legend(loc="best")

    fig.suptitle(f"Plot D: Correction impact - {roi}")
    return savefig(fig, outdir / f"plot_D_correction_impact_{roi}.png")


def plot_E(df, outdir, roi, limits):
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df["elapsed_hours"], df[f"{roi}_sig_raw"], linewidth=0.8, label="Sig Raw / 465 calcium")
    ax.plot(df["elapsed_hours"], df[f"{roi}_uv_raw"], linewidth=0.8, label="UV Raw / 405 isosbestic")
    ax.set_title(f"Plot E: 465 and 405 raw channels on the same graph - {roi}")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Signal (volts / a.u.)")
    set_xlim_hours(ax, limits)
    set_ylim(ax, limits, "raw_both")
    ax.legend(loc="best")
    return savefig(fig, outdir / f"plot_E_both_signals_same_graph_{roi}.png")


def plot_F(df, outdir, roi, limits):
    fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True)
    axes[0].plot(df["elapsed_hours"], df[f"{roi}_sig_raw"], linewidth=0.8)
    axes[0].set_title("465 nm calcium-dependent signal")
    axes[0].set_ylabel("465 raw")
    set_ylim(axes[0], limits, "sig_raw")

    axes[1].plot(df["elapsed_hours"], df[f"{roi}_uv_raw"], linewidth=0.8)
    axes[1].set_title("405 nm isosbestic/control signal")
    axes[1].set_ylabel("405 raw")
    axes[1].set_xlabel("Time (hours)")
    set_ylim(axes[1], limits, "uv_raw")
    set_xlim_hours(axes[1], limits)

    fig.suptitle(f"Plot F: Raw signals independently - {roi}")
    return savefig(fig, outdir / f"plot_F_signals_independent_{roi}.png")


def plot_G(df, outdir, roi, limits):
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    axes[0].plot(df["elapsed_hours"], df[f"{roi}_deltaF"], linewidth=0.8)
    axes[0].set_title("Delta-F corrected signal")
    axes[0].set_ylabel("Delta-F")
    set_ylim(axes[0], limits, "deltaF")

    axes[1].plot(df["elapsed_hours"], df[f"{roi}_dFF"], linewidth=0.8)
    axes[1].set_title("Main DF/F0 plot")
    axes[1].set_ylabel("DF/F0")
    set_ylim(axes[1], limits, "dFF")

    axes[2].plot(df["elapsed_hours"], df[f"{roi}_z_dFF"], linewidth=0.8)
    axes[2].set_title("z-scored DF/F0")
    axes[2].set_ylabel("z-DF/F0")
    axes[2].set_xlabel("Time (hours)")
    set_ylim(axes[2], limits, "z_dFF")
    set_xlim_hours(axes[2], limits)

    fig.suptitle(f"Plot G: Lowell-style processed outputs - {roi}")
    return savefig(fig, outdir / f"plot_G_lowell_deltaF_DF_F0_z_{roi}.png")


def plot_H(df, outdir, roi, limits):
    fig, ax1 = plt.subplots(figsize=(16, 6))
    ax2 = ax1.twinx()
    l1, = ax1.plot(df["elapsed_hours"], df[f"{roi}_deltaF"], linewidth=0.8, label="Delta-F")
    l2, = ax2.plot(df["elapsed_hours"], df[f"{roi}_dFF"], linewidth=0.8, label="DF/F0")
    ax1.set_title(f"Plot H: Delta-F and dFF on the same graph - {roi}")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Delta-F")
    ax2.set_ylabel("DF/F0")
    set_ylim(ax1, limits, "deltaF")
    set_ylim(ax2, limits, "dFF")
    set_xlim_hours(ax1, limits)
    ax1.legend([l1, l2], [l1.get_label(), l2.get_label()], loc="best")
    return savefig(fig, outdir / f"plot_H_deltaF_and_dFF_same_graph_{roi}.png")


def make_interactive(df, outdir, roi, limits):
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.04,
        subplot_titles=(
            "Raw 465 and 405 together",
            "Correction check: 465 signal and fitted 405",
            "Delta-F corrected signal",
            "DF/F0",
            "z-scored dFF",
        ),
    )
    x = df["elapsed_hours"]
    fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_raw"], mode="lines", name="UV Raw / 405"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_sig_raw"], mode="lines", name="Sig Raw / 465"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_uv_fit"], mode="lines", name="UV Fit / fitted 405"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_deltaF"], mode="lines", name="Delta-F"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_dFF"], mode="lines", name="DF/F0"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=df[f"{roi}_z_dFF"], mode="lines", name="z-DF/F0"), row=5, col=1)

    fig.update_yaxes(title_text="Raw signal", row=1, col=1)
    fig.update_yaxes(title_text="Fit check", row=2, col=1)
    fig.update_yaxes(title_text="Delta-F", row=3, col=1)
    fig.update_yaxes(title_text="DF/F0", row=4, col=1)
    fig.update_yaxes(title_text="z-DF/F0", row=5, col=1)

    if limits is not None:
        fig.update_yaxes(range=limits["raw_both"], row=1, col=1)
        fig.update_yaxes(range=limits["fit_check"], row=2, col=1)
        fig.update_yaxes(range=limits["deltaF"], row=3, col=1)
        fig.update_yaxes(range=limits["dFF"], row=4, col=1)
        fig.update_yaxes(range=limits["z_dFF"], row=5, col=1)

    for row in range(1, 6):
        fig.update_xaxes(title_text="Time (hours)", matches=None, row=row, col=1)
        if limits is not None:
            fig.update_xaxes(range=limits["x_hours"], row=row, col=1)

    fig.update_layout(
        title=f"Interactive Lowell-style photometry outputs - {roi} - matched scales - no event data used",
        height=1100, width=1500, hovermode="x unified",
        legend=dict(x=1.02, y=1.0), margin=dict(l=80, r=260, t=90, b=60),
    )
    path = outdir / f"interactive_lowell_style_no_events_{roi}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def write_readme(outdir, roi, ppd_path, header, fs, processed, limits, scale_mode):
    path = outdir / "README_NO_EVENTS_MATCHED_SCALES.md"
    path.write_text(f"""# Lowell-style no-event pyPhotometry analysis with matched axes

## Input file

`{ppd_path.name}`

No feeding/event timestamp data was required or used.

## Matched scale behavior

This script was run with:

```text
scale_mode = {scale_mode}
```

Every recording processed in the same command uses the same x-axis and y-axis ranges for the matching plots.
That means Plot G from recording 1 can be visually compared to Plot G from recording 2 without the axes changing.

Matched y-axis groups:

- raw_both: Plot A, Plot E, interactive raw row, and Plot D raw panel
- uv_raw: 405-only raw panels in Plot B and Plot F
- sig_raw: 465-only raw panels in Plot B and Plot F
- fit_check: correction check / fitted 405 panel
- deltaF: Plot D bottom, Plot G top, Plot H left axis
- dFF: Plot G middle, Plot H right axis
- z_dFF: Plot G bottom

## How to analyze two recordings with identical scales

```bash
python ana_same_scales.py recording1.ppd recording2.ppd --roi BLA --scale-mode full
```

If one recording has a very large movement artifact spike and the useful signal looks compressed, use robust scaling:

```bash
python ana_same_scales.py recording1.ppd recording2.ppd --roi BLA --scale-mode robust
```

## Channel extraction

Assumed pyPhotometry 2EX_1EM_pulsed layout:

```text
{roi}_sig_raw = 465_on - 465_dark
{roi}_uv_raw  = 405_on - 405_dark
```

## Processing formulas

Median + low-pass filter:

```text
{roi}_sig_median_filtered = median-filtered {roi}_sig_raw
{roi}_uv_median_filtered  = median-filtered {roi}_uv_raw
{roi}_sig_filt_10Hz_lowpass = 10 Hz low-pass filtered {roi}_sig_median_filtered
{roi}_uv_filt_10Hz_lowpass  = 10 Hz low-pass filtered {roi}_uv_median_filtered
```

Motion correction:

```text
{roi}_uv_fit = slope × {roi}_uv_filt_10Hz_lowpass + intercept
{roi}_deltaF = {roi}_sig_filt_10Hz_lowpass - {roi}_uv_fit
```

This uses one global linear best fit to the 405 nm signal, matching the Lowell/Douglass paper method.

F0:

```text
F0(t) = 0.001 Hz low-pass filtered {roi}_sig_filt_10Hz_lowpass across the whole session
F0 median = {processed['F0']}
```

dFF:

```text
{roi}_dFF = {roi}_deltaF / F0(t)
```

z-dFF:

```text
{roi}_z_dFF = ({roi}_dFF - mean({roi}_dFF)) / SD({roi}_dFF)
```

## Main dFF graph

Use:

```text
plot_G_lowell_deltaF_DF_F0_z_{roi}.png
```

The middle panel is the main Lowell-style DF/F0 trace. The column is still named `{roi}_dFF` for app compatibility, but it is a ratio, not percent.

## Axis limits used

```json
{json.dumps(limits, indent=2)}
```

## Analysis settings

```json
{json.dumps({
    'sampling_rate_hz': fs,
    'photometry_lowpass_hz': LOWPASS_HZ,
    'filter_order': FILTER_ORDER,
    'median_filter_sec': MEDIAN_FILTER_SEC,
    'long_term_smooth_sec': LONG_TERM_SMOOTH_SEC,
    'motion_correction': 'global_linear_best_fit_405_to_465',
    'baseline_method': 'lowpass_filtered_465_signal_whole_session',
    'baseline_lowpass_hz': F0_LOWPASS_HZ,
    'F0': processed['F0'],
    'event_data_used': False,
}, indent=2)}
```
""")
    return path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def process_one_recording(ppd_path: Path, roi: str, save_full: bool):
    header, fs, volts = read_ppd(ppd_path)
    uv_raw, sig_raw = extract_channels(volts)
    processed = process_photometry(uv_raw, sig_raw, fs)
    time_sec = np.arange(len(sig_raw), dtype=float) / fs
    df_full = make_dataframe(time_sec, uv_raw, sig_raw, processed, roi)
    df_plot = downsample(df_full, fs, EXPORT_HZ)
    return {
        "ppd_path": ppd_path,
        "header": header,
        "fs": fs,
        "processed": processed,
        "df_full": df_full if save_full else None,
        "df_plot": df_plot,
    }


def save_one_recording_outputs(recording, outdir: Path, roi: str, limits, scale_mode: str, save_full: bool):
    outdir.mkdir(parents=True, exist_ok=True)
    df_plot = recording["df_plot"]
    df_full = recording["df_full"]
    ppd_path = recording["ppd_path"]
    header = recording["header"]
    fs = recording["fs"]
    processed = recording["processed"]

    df_plot.to_csv(outdir / f"processed_jones_style_downsampled_{EXPORT_HZ:g}Hz.csv", index=False)
    if save_full and df_full is not None:
        df_full.to_csv(outdir / "processed_jones_style_full_20Hz.csv", index=False)

    settings = {
        "input_file": str(ppd_path),
        "subject_ID": header.get("subject_ID", ""),
        "sampling_rate_hz": fs,
        "event_data_used": False,
        "scale_mode": scale_mode,
        "shared_axis_limits_applied": limits is not None,
        "lowpass_hz": LOWPASS_HZ,
        "filter_order": FILTER_ORDER,
        "median_filter_sec": MEDIAN_FILTER_SEC,
        "long_term_smooth_sec": LONG_TERM_SMOOTH_SEC,
        "motion_correction": "global_linear_best_fit_405_to_465",
        "baseline_method": "lowpass_filtered_465_signal_whole_session",
        "baseline_lowpass_hz": F0_LOWPASS_HZ,
        "F0": processed["F0"],
        "deltaF_formula": f"{roi}_deltaF = {roi}_sig_raw - {roi}_uv_fit",
        "dFF_formula": f"{roi}_dFF = {roi}_deltaF / F0_trace",
    }
    (outdir / "ANALYSIS_SETTINGS_NO_EVENTS_MATCHED_SCALES.json").write_text(json.dumps(settings, indent=2))
    (outdir / "PPD_HEADER.json").write_text(json.dumps(header, indent=2))

    paths = [
        plot_A(df_plot, outdir, roi, limits),
        plot_B(df_plot, outdir, roi, limits),
        plot_C(df_plot, outdir, roi, limits),
        plot_D(df_plot, outdir, roi, limits),
        plot_E(df_plot, outdir, roi, limits),
        plot_F(df_plot, outdir, roi, limits),
        plot_G(df_plot, outdir, roi, limits),
        plot_H(df_plot, outdir, roi, limits),
        make_interactive(df_plot, outdir, roi, limits),
        write_readme(outdir, roi, ppd_path, header, fs, processed, limits, scale_mode),
    ]

    try:
        script_copy = outdir / "ana_same_scales.py"
        script_copy.write_text(Path(__file__).read_text())
        paths.append(script_copy)
    except Exception:
        pass

    return paths


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lowell-style .ppd analysis with no events and matched axes across recordings."
    )
    parser.add_argument(
        "ppd",
        nargs="*",
        default=None,
        help="One or more .ppd files. If omitted, DEFAULT_PPD_PATHS is used.",
    )
    parser.add_argument("--roi", default=DEFAULT_ROI_NAME, help="ROI name for columns/plots")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Parent output folder. Default: jones_style_matched_scales_output",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["full", "robust", "none"],
        default="full",
        help=(
            "full = shared min/max limits across all recordings; "
            "robust = shared 0.5-99.5 percentile limits; "
            "none = old auto-scaling behavior."
        ),
    )
    parser.add_argument("--save-full", action="store_true", help="Also save full 20 Hz processed CSV; can be large")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.ppd:
        ppd_paths = [find_ppd_file(Path(p)) for p in args.ppd]
    else:
        ppd_paths = [find_ppd_file(p) for p in DEFAULT_PPD_PATHS]

    roi = args.roi
    parent_outdir = Path(args.outdir) if args.outdir else Path("jones_style_matched_scales_output")
    parent_outdir.mkdir(parents=True, exist_ok=True)

    print("Processing recordings first so shared axis limits can be calculated...")
    recordings = [process_one_recording(p, roi, save_full=args.save_full) for p in ppd_paths]

    limits = compute_axis_limits(recordings, roi, args.scale_mode)
    if limits is not None:
        (parent_outdir / "GLOBAL_AXIS_LIMITS_USED_FOR_ALL_RECORDINGS.json").write_text(json.dumps(limits, indent=2))

    output_folders = []
    for rec in recordings:
        rec_outdir = parent_outdir / safe_stem(rec["ppd_path"])
        save_one_recording_outputs(rec, rec_outdir, roi, limits, args.scale_mode, save_full=args.save_full)
        output_folders.append(rec_outdir)

    # Copy script into parent folder.
    try:
        shutil.copy2(Path(__file__), parent_outdir / "ana_same_scales.py")
    except Exception:
        pass

    # Zip the whole parent output folder.
    zip_path = parent_outdir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in parent_outdir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(parent_outdir)))

    print("Done.")
    print(f"Input files: {len(ppd_paths)}")
    print(f"Scale mode: {args.scale_mode}")
    print(f"Parent output folder: {parent_outdir.resolve()}")
    print(f"ZIP file: {zip_path.resolve()}")
    if limits is not None:
        print("Global axis limits saved to GLOBAL_AXIS_LIMITS_USED_FOR_ALL_RECORDINGS.json")
    for folder in output_folders:
        print(f"Output folder: {folder.resolve()}")
        print(f"  Main DF/F0 plot: {folder / f'plot_G_lowell_deltaF_DF_F0_z_{roi}.png'}")
        print(f"  Interactive HTML: {folder / f'interactive_lowell_style_no_events_{roi}.html'}")


if __name__ == "__main__":
    main()
