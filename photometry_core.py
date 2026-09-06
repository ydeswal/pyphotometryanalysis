#!/usr/bin/env python3
"""
photometry_core.py
==================

Single source of truth for all photometry maths used by this project.

Before this module existed, `ana.py` (used for .ppd files) and `app.py` (used for
.csv files) implemented *different* pipelines, so the same animal analysed by two
routes gave different dF/F values and different z-scores. Everything now routes
through here.

Reference methods
-----------------
* pyPhotometry .ppd binary layout follows the official spec:
  https://pyphotometry.readthedocs.io/en/latest/user-guide/importing-data/
* Isosbestic fitting / dF/F recommendations follow Keevers & Jean-Richard-dit-Bressel
  (2025) Neurophotonics 12:025003 - low-pass filter, IRLS (robust) regression of the
  405 control onto the 465 signal, then dF/F rather than bare dF.
* Long-timescale (multi-hour / multi-day) F0 and 30-min running means follow the
  Lowell-lab style used for slow homeostatic signals.
* Baseline-window z-scoring (mean/SD taken from a defined baseline epoch rather
  than the whole session) is the standard for event-locked analysis.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import median_filter as _nd_median_filter

# Samples per filtering block. Chosen so the float64 working copies inside
# sosfiltfilt stay well under ~100 MB regardless of how long the recording is.
_FILTER_BLOCK = 2_000_000

__all__ = [
    "read_ppd",
    "decimate_after_lowpass",
    "median_filter_trace",
    "lowpass",
    "fit_control_to_signal",
    "compute_f0",
    "zscore",
    "rolling_mean_by_time",
    "crop_and_rezero",
    "process_photometry",
    "DEFAULTS",
]


# =============================================================================
# DEFAULT ANALYSIS SETTINGS
# =============================================================================

DEFAULTS = {
    "median_filter_sec": 0.0,      # 0 disables; a short median filter kills sharp artefacts
    "lowpass_hz": 10.0,            # photometry trace low-pass
    "filter_order": 3,
    "fit_method": "irls",          # "irls" (robust, recommended) or "ols"
    "irls_tuning": 1.345,          # Huber tuning constant
    "f0_method": "lowpass",        # "lowpass" | "rolling_percentile" | "percentile" | "median"
    "f0_lowpass_hz": 0.001,        # Lowell-style very slow baseline
    "f0_percentile": 10.0,
    "f0_window_sec": 1800.0,       # window for rolling_percentile
    "dff_units": "ratio",          # "ratio" or "percent"
    "zscore_mode": "session",      # "session" | "baseline" | "robust"
    "long_term_smooth_sec": 1800.0,
    "keep_intermediates": False,
}


# =============================================================================
# .PPD READING  (mode-aware, spec-compliant)
# =============================================================================

def _header_int(header: dict, *keys, default=None):
    """pyPhotometry has used different header key names across versions."""
    for k in keys:
        if k in header and header[k] is not None:
            return int(header[k])
    if default is not None:
        return int(default)
    raise KeyError(f"None of {keys} found in .ppd header. Header keys: {list(header)}")


def _parse_version(version_string) -> tuple:
    try:
        parts = str(version_string).strip().split(".")
        return tuple(int("".join(ch for ch in p if ch.isdigit()) or 0) for p in parts[:3])
    except Exception:
        return (0, 0, 0)


def read_ppd(ppd_path, dtype=np.float32, keep_components=False,
             with_time=False) -> dict:
    """
    Read a pyPhotometry .ppd file into named signals.

    THIS IS THE MOST IMPORTANT CORRECTION IN THIS PROJECT.

    The .ppd data block is a flat stream of little-endian uint16. In each 2-byte
    word the top 15 bits are an analog sample and the bottom bit is a digital
    sample. How those words are interleaved depends on acquisition mode:

    Pulsed modes, pyPhotometry >= v1.1  (4 words per frame):
        word 0 : analog channel 1, LED ON
        word 1 : analog channel 1, LED OFF baseline
        word 2 : analog channel 2, LED ON
        word 3 : analog channel 2, LED OFF baseline
      -> analog_1 = w0 - w1   (e.g. 465 GCaMP, dark-subtracted)
      -> analog_2 = w2 - w3   (e.g. 405 isosbestic, dark-subtracted)
      -> digital_1 = LSB of w0, digital_2 = LSB of w2

    Continuous modes, and ALL pulsed files written before v1.1 (2 words/frame):
        word 0 : analog channel 1 (already baseline-subtracted on-board)
        word 1 : analog channel 2
      -> analog_1 = w0, analog_2 = w1
      -> digital_1 = LSB of w0, digital_2 = LSB of w1

    The previous code always reshaped to 4 columns, because it computed
    `n_analog_signals + n_digital_signals` = 2 + 2 = 4. That happens to be right
    for new-format pulsed files but is silently, badly wrong for continuous-mode
    or pre-v1.1 files: it interleaves two consecutive timepoints and then
    subtracts 405 from 465 twice, so "signal" and "isosbestic" become the same
    quantity and every downstream number is meaningless.

    Returns a dict with analog_1, analog_2, digital_1, digital_2, time_sec,
    sampling_rate, header, frame_layout, plus the raw LED-on/LED-off components
    when the file is new-format pulsed.
    """
    ppd_path = Path(ppd_path)
    dtype = np.dtype(dtype)
    with open(ppd_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        header = json.loads(f.read(header_size).decode("utf-8"))
        raw = np.frombuffer(f.read(), dtype="<u2")

    fs = float(header["sampling_rate"])
    mode = str(header.get("mode", "")).strip()
    version = _parse_version(header.get("version", "0.0.0"))

    n_analog = _header_int(
        header, "n_analog_channels", "n_analog_signals", default=2
    )

    vpd_raw = header.get("volts_per_division", [1.0])
    if np.isscalar(vpd_raw):
        vpd = [float(vpd_raw)] * n_analog
    else:
        vpd = [float(v) for v in vpd_raw]
    while len(vpd) < n_analog:
        vpd.append(vpd[-1])

    # Do NOT convert the whole stream at once. For a 48 h recording the data
    # block is 90 million uint16 words; `(raw >> 1).astype(np.float64)` would
    # allocate a single 719 MB array before a single channel is extracted.
    # Channels are sliced out of the uint16 buffer first and converted
    # individually, in float32 - a 15-bit ADC value needs 15 bits of mantissa
    # and float32 has 24, so nothing is lost.

    # ---- decide the frame layout -------------------------------------------
    is_pulsed = "pulsed" in mode.lower()
    new_format = version >= (1, 1)
    words_per_frame = 2 * n_analog if (is_pulsed and new_format) else n_analog

    # Fall back gracefully if the declared layout does not divide the data.
    if raw.size % words_per_frame != 0:
        alt = n_analog if words_per_frame != n_analog else 2 * n_analog
        if raw.size % alt == 0:
            warnings.warn(
                f"{ppd_path.name}: data length {raw.size} is not divisible by the "
                f"expected {words_per_frame} words/frame; falling back to {alt}."
            )
            words_per_frame = alt
        else:
            raise ValueError(
                f"{ppd_path.name}: data length {raw.size} is divisible by neither "
                f"{words_per_frame} nor {alt} words per frame."
            )

    out = {
        "header": header,
        "sampling_rate": fs,
        "mode": mode,
        "version": header.get("version", ""),
        "subject_ID": header.get("subject_ID", ""),
        "n_analog_channels": n_analog,
        "volts_per_division": vpd,
    }

    def _analog(offset):
        """Slice one interleaved stream and convert just that slice."""
        return (raw[offset::words_per_frame] >> 1).astype(dtype)

    def _digital(offset):
        return (raw[offset::words_per_frame] & 1).astype(np.uint8)

    if words_per_frame == 2 * n_analog:
        # New-format pulsed: LED-on and LED-off stored separately.
        out["frame_layout"] = "pulsed_v1.1_led_on_off"
        for ch in range(n_analog):
            on = _analog(2 * ch)
            off = _analog(2 * ch + 1)
            n = min(len(on), len(off))
            on, off = on[:n], off[:n]
            on *= vpd[ch]
            off *= vpd[ch]
            if keep_components:
                out[f"analog_{ch + 1}_raw_LED_on"] = on.copy()
                out[f"analog_{ch + 1}_raw_baseline"] = off.copy()
            on -= off          # in place; avoids a third full-length array
            out[f"analog_{ch + 1}"] = on
            out[f"digital_{ch + 1}"] = _digital(2 * ch)[:n]
            del off
    else:
        # Continuous, or pulsed written before v1.1 (already dark-subtracted).
        out["frame_layout"] = "one_word_per_channel"
        for ch in range(n_analog):
            sig = _analog(ch)
            sig *= vpd[ch]
            out[f"analog_{ch + 1}"] = sig
            out[f"digital_{ch + 1}"] = _digital(ch)[: len(sig)]

    del raw
    n_samples = len(out["analog_1"])

    out["n_samples"] = n_samples
    out["duration_sec"] = n_samples / fs
    # The time axis is NOT materialised by default. At 48 h it is 180 MB of
    # float64 - 45% of everything this function retains - and it is trivially
    # reconstructible as arange(n)/fs. Callers that need it ask for it.
    # float64 is required: at 48 h the values reach 172,800 and float32 would
    # quantise them to about 0.01 s.
    if with_time:
        out["time_sec"] = np.arange(n_samples, dtype=np.float64) / fs
    return out


def digital_pulse_times(digital: np.ndarray, fs: float) -> np.ndarray:
    """Rising-edge times (seconds) of a digital channel. Used for lickometer TTL."""
    d = np.asarray(digital).astype(np.int8)
    if d.size < 2:
        return np.array([], dtype=float)
    rising = np.flatnonzero(np.diff(d) == 1) + 1
    return rising.astype(float) / float(fs)


# =============================================================================
# FILTERS
# =============================================================================

def median_filter_trace(values: np.ndarray, fs: float, window_sec: float) -> np.ndarray:
    """
    Median filter with edge reflection.

    Uses scipy.ndimage.median_filter rather than scipy.signal.medfilt. medfilt is
    O(n*k) and zero-pads the edges: on a 48 h recording at 130 Hz with a 1 s
    kernel that is ~3e9 operations and it corrupts the first and last half-kernel
    of the trace. ndimage uses a proper rolling histogram and reflects the edges.
    """
    # Preserve the input dtype. Casting float32 to float64 here doubled memory
    # on every long recording for no numerical benefit: a 15-bit ADC value fits
    # comfortably in float32's 24-bit mantissa.
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    if window_sec is None or window_sec <= 0:
        return values.copy()
    k = int(round(float(window_sec) * float(fs)))
    if k < 3:
        return values.copy()
    if k % 2 == 0:
        k += 1
    if k >= values.size:
        return np.full_like(values, float(np.nanmedian(values)))
    return _nd_median_filter(values, size=k, mode="reflect")


def lowpass(values, fs, cutoff_hz, order=3):
    """
    Zero-phase Butterworth low-pass.

    For very low cutoffs (e.g. the 0.001 Hz baseline filter) a Butterworth
    designed at the full sampling rate sits pathologically close to the unit
    circle and loses precision. Below a normalised cutoff of 1e-3 we decimate,
    filter on the decimated series, then interpolate back. This is both far more
    numerically stable and dramatically faster.
    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    in_dtype = values.dtype
    fs = float(fs)
    if cutoff_hz is None or cutoff_hz <= 0 or values.size < 12:
        return values.copy()

    nyquist = fs / 2.0
    cutoff = min(float(cutoff_hz), nyquist * 0.99)
    wn = cutoff / nyquist

    if wn < 1e-3:
        target_fs = max(cutoff * 20.0, 1e-6)
        step = max(int(np.floor(fs / target_fs)), 1)
        if step > 1 and values.size // step >= 12:
            idx = np.arange(0, values.size, step)
            small = values[idx]
            small_fs = fs / step
            filt_small = lowpass(small, small_fs, cutoff, order=order)
            return np.interp(
                np.arange(values.size, dtype=float), idx.astype(float), filt_small
            )

    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    padlen = 3 * (2 * order)
    if values.size <= padlen:
        warnings.warn("Trace too short for zero-phase filtering; returning unfiltered.")
        return values.copy()

    # Long traces are filtered in overlapping blocks so peak memory does not
    # scale with recording length. sosfiltfilt promotes its input to float64 and
    # allocates several working copies; on a 48 h recording at 130 Hz that is
    # over a gigabyte for a single channel, which is what made multi-day files
    # crash. Each block is filtered with a generous margin on both sides and
    # only the interior is kept, so the result is identical to filtering the
    # whole array at once.
    if values.size > _FILTER_BLOCK:
        margin = max(int(round(20.0 * fs / max(cutoff, 1e-9))), 10 * padlen)
        margin = min(margin, values.size // 2)
        out = np.empty_like(values)
        step = _FILTER_BLOCK
        for start in range(0, values.size, step):
            stop = min(start + step, values.size)
            lo = max(0, start - margin)
            hi = min(values.size, stop + margin)
            try:
                seg = sosfiltfilt(sos, values[lo:hi])
            except ValueError:
                seg = values[lo:hi].astype(np.float64)
            out[start:stop] = seg[start - lo: stop - lo].astype(in_dtype, copy=False)
            del seg
        return out

    try:
        # sosfiltfilt promotes to float64 and makes several working copies. On a
        # 48 h trace that is over a gigabyte per channel, so the result is cast
        # straight back down.
        return sosfiltfilt(sos, values).astype(in_dtype, copy=False)
    except ValueError:
        warnings.warn("Low-pass filter failed; returning unfiltered trace.")
        return values.copy()


# =============================================================================
# ISOSBESTIC (405 -> 465) FITTING
# =============================================================================

def _ols_fit(x, y):
    A = np.column_stack([x, np.ones_like(x)])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _irls_fit(x, y, tuning=1.345, max_iter=50, tol=1e-8):
    """
    Huber iteratively-reweighted least squares.

    Keevers & Jean-Richard-dit-Bressel (2025) show IRLS beats OLS for fitting the
    isosbestic control onto the signal, because real calcium transients are
    genuine divergences between the two channels. OLS treats them as error and
    drags the fit toward them, so part of the real signal gets subtracted away.
    IRLS down-weights those points and fits the artefact component instead.
    """
    A = np.column_stack([x, np.ones_like(x)])
    slope, intercept = _ols_fit(x, y)
    beta = np.array([slope, intercept], dtype=float)

    for _ in range(max_iter):
        resid = y - A @ beta
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            break
        u = resid / (tuning * scale)
        w = np.where(np.abs(u) <= 1.0, 1.0, 1.0 / np.maximum(np.abs(u), 1e-12))
        sw = np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return float(beta[0]), float(beta[1])


def fit_control_to_signal(control, signal, method="irls", tuning=1.345):
    """
    Scale the 405 control onto the 465 signal.

    Returns (fitted_control, slope, intercept). fitted = slope*control + intercept.
    """
    control = np.asarray(control, dtype=float)
    signal = np.asarray(signal, dtype=float)
    good = np.isfinite(control) & np.isfinite(signal)
    if good.sum() < 3:
        raise ValueError("Not enough finite samples to fit 405 onto 465.")

    x, y = control[good], signal[good]
    if method == "ols":
        slope, intercept = _ols_fit(x, y)
    else:
        slope, intercept = _irls_fit(x, y, tuning=tuning)

    return slope * control + intercept, slope, intercept


# =============================================================================
# BASELINE (F0)
# =============================================================================

def compute_f0(signal, fs, method="lowpass", lowpass_hz=0.001,
               percentile=10.0, window_sec=1800.0):
    """
    Estimate baseline fluorescence F0 from the SIGNAL channel.

    F0 must be derived from the 465 signal (or its slow envelope). The previous
    CSV path took the 10th percentile of the *raw 405 isosbestic* channel as F0,
    which is a different physical quantity on a different scale - it made every
    dF/F value from the CSV route wrong in magnitude.

    Methods
    -------
    "lowpass"            : very slow low-pass of the signal (Lowell-style, good
                           for multi-hour recordings; tracks photobleaching).
    "rolling_percentile" : running low percentile in a moving window; robust to
                           long positive excursions.
    "percentile"         : single session-wide percentile (classic short sessions).
    "median"             : session median.
    """
    signal = np.asarray(signal)
    if not np.issubdtype(signal.dtype, np.floating):
        signal = signal.astype(np.float32)

    if method == "lowpass":
        f0 = lowpass(signal, fs, lowpass_hz)
    elif method == "rolling_percentile":
        win = max(int(round(float(window_sec) * float(fs))), 3)
        f0 = (
            pd.Series(signal)
            .rolling(win, center=True, min_periods=max(3, win // 10))
            .quantile(percentile / 100.0)
            .bfill().ffill().to_numpy()
        )
    elif method == "percentile":
        f0 = np.full_like(signal, float(np.nanpercentile(signal, percentile)))
    elif method == "median":
        f0 = np.full_like(signal, float(np.nanmedian(signal)))
    else:
        raise ValueError(f"Unknown F0 method: {method}")

    # Guard against dividing by ~0. Anything that small is not a real baseline.
    scale = np.nanmedian(np.abs(f0))
    eps = scale * 1e-9 if np.isfinite(scale) and scale > 0 else 1e-12
    return np.where(np.isfinite(f0) & (np.abs(f0) > eps), f0, np.nan)


# =============================================================================
# Z-SCORING
# =============================================================================

def zscore(values, baseline_mask=None, robust=False):
    """
    z = (x - centre) / spread

    baseline_mask
        Boolean mask selecting the epoch used to define centre and spread. This
        is the correct approach for event-locked analysis: the resulting z-score
        means "standard deviations away from how this animal looked at rest".
        With mask=None the whole session defines the statistics, which is fine
        for describing the shape of one long trace but makes values from
        different sessions non-comparable, since each session is forced to
        mean 0 / SD 1 by construction.

    robust
        Use median and 1.4826*MAD instead of mean and SD, so a few large
        transients or a movement artefact cannot inflate the denominator.
    """
    values = np.asarray(values, dtype=float)
    ref = values if baseline_mask is None else values[np.asarray(baseline_mask, dtype=bool)]
    ref = ref[np.isfinite(ref)]
    if ref.size < 2:
        return np.full_like(values, np.nan)

    if robust:
        centre = float(np.median(ref))
        spread = 1.4826 * float(np.median(np.abs(ref - centre)))
    else:
        centre = float(np.mean(ref))
        spread = float(np.std(ref, ddof=1))

    if not np.isfinite(spread) or spread <= 0:
        return np.full_like(values, np.nan)
    return (values - centre) / spread


def rolling_mean_by_time(time_sec, values, window_sec=1800.0, centered=True):
    """Time-aware rolling mean; correct even with irregular sampling or gaps."""
    time_sec = np.asarray(time_sec, dtype=float)
    values = np.asarray(values, dtype=float)
    if time_sec.size == 0:
        return values.copy()

    order = np.argsort(time_sec)
    s = pd.Series(values[order], index=pd.to_timedelta(time_sec[order], unit="s"))
    smooth = s.rolling(
        window=pd.Timedelta(seconds=max(float(window_sec), 1.0)),
        center=bool(centered),
        min_periods=1,
    ).mean().to_numpy()

    out = np.empty_like(smooth, dtype=float)
    out[order] = smooth
    return out


# =============================================================================
# CROPPING WITH TIME RE-ZERO
# =============================================================================

def crop_and_rezero(df, start_sec=None, end_sec=None, rezero=True,
                    time_col="time_sec"):
    """
    Keep [start_sec, end_sec] and optionally shift time so the crop start is t=0.

    Crop a 48 h recording at start_sec = 86400 and the sample that was at hour 24
    becomes hour 0, hour 25 becomes hour 1, and so on.

    The offset subtracted is exactly `start_sec`, not the timestamp of the first
    surviving sample. If sampling starts a fraction of a second after the
    requested boundary, using the first sample would silently shift the axis by
    that fraction; using the requested boundary makes "crop at 24 h" land on
    exactly 00:00:00.

    The original timeline is preserved in `original_time_sec` so nothing is lost
    and events can still be mapped back.

    Returns (cropped_dataframe, offset_seconds).
    """
    out = df.copy()
    if time_col not in out.columns:
        raise ValueError(f"Column '{time_col}' not found for cropping.")

    t = pd.to_numeric(out[time_col], errors="coerce")
    mask = np.ones(len(out), dtype=bool)
    if start_sec is not None:
        mask &= (t >= float(start_sec)).to_numpy()
    if end_sec is not None:
        mask &= (t <= float(end_sec)).to_numpy()

    out = out.loc[mask].reset_index(drop=True)
    if out.empty:
        return out, 0.0

    offset = float(start_sec) if (rezero and start_sec is not None) else 0.0
    if offset:
        if "original_time_sec" not in out.columns:
            out["original_time_sec"] = out[time_col].to_numpy(dtype=float)
        out[time_col] = out[time_col].to_numpy(dtype=float) - offset
        if "elapsed_hours" in out.columns:
            out["elapsed_hours"] = out[time_col] / 3600.0
    return out, offset


# =============================================================================
# FULL PIPELINE
# =============================================================================

def decimate_after_lowpass(values, factor):
    """
    Reduce the sample rate by an integer factor by averaging within bins.

    Safe only AFTER low-pass filtering: the 10 Hz filter has already removed
    everything above 10 Hz, so sampling faster than ~20 Hz stores no additional
    information. Dropping from 130 Hz to 20 Hz cuts memory more than sixfold
    while leaving the analysed signal untouched.
    """
    values = np.asarray(values)
    factor = int(max(1, factor))
    if factor == 1:
        return values
    n = (len(values) // factor) * factor
    if n == 0:
        return values
    return values[:n].reshape(-1, factor).mean(axis=1)


def process_photometry(signal_raw, control_raw, fs, settings=None,
                       baseline_mask=None, analysis_hz=None, consume=False):
    """
    Run the full corrected pipeline on one recording.

    signal_raw  : 465 nm / biosensor channel (dark-subtracted volts)
    control_raw : 405 nm / isosbestic channel (dark-subtracted volts)

    Order of operations:
      1. median filter (optional) to remove sharp artefacts
      2. low-pass filter both channels at the same cutoff
      3. fit the filtered control onto the filtered signal (IRLS by default)
      4. deltaF = filtered signal - fitted control      [both filtered]
      5. F0 from the SIGNAL channel
      6. dF/F = deltaF / F0
      7. z-score of dF/F (session, baseline-window, or robust)
    """
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update(settings)

    signal_raw = np.asarray(signal_raw)
    control_raw = np.asarray(control_raw)

    keep = cfg.get("keep_intermediates", False)

    # Channels are taken through median + low-pass ONE AT A TIME, and each
    # intermediate is released as soon as it has been consumed. Doing both
    # channels stage by stage meant six full-rate arrays were alive at once;
    # at 48 h and 130 Hz that is 540 MB of signal before any working copies.
    #
    # When consume=True the caller has promised it holds no other reference to
    # the input arrays, so they can be freed here too.
    def _prep(raw):
        med = median_filter_trace(raw, fs, cfg["median_filter_sec"])
        filt = lowpass(med, fs, cfg["lowpass_hz"], cfg["filter_order"])
        return (med, filt) if keep else (None, filt)

    sig_med, sig_filt = _prep(signal_raw)
    if consume:
        del signal_raw
        signal_raw = None
    ctl_med, ctl_filt = _prep(control_raw)
    if consume:
        del control_raw
        control_raw = None

    # Optional decimation, applied only after low-pass filtering.
    decim = 1
    if analysis_hz and analysis_hz > 0 and fs > analysis_hz:
        nyquist_needed = 2.5 * cfg["lowpass_hz"]
        target = max(float(analysis_hz), nyquist_needed)
        decim = int(max(1, np.floor(fs / target)))
        if decim > 1:
            sig_filt = decimate_after_lowpass(sig_filt, decim)
            ctl_filt = decimate_after_lowpass(ctl_filt, decim)
            fs = fs / decim

    # Step 4: BOTH sides filtered. The old CSV path fitted on filtered traces but
    # then subtracted from the RAW traces, reinjecting all the high-frequency
    # noise it had just removed and making dF/F far noisier than intended.
    ctl_fit, slope, intercept = fit_control_to_signal(
        ctl_filt, sig_filt, method=cfg["fit_method"], tuning=cfg["irls_tuning"]
    )
    delta_f = sig_filt - ctl_fit

    f0 = compute_f0(
        sig_filt, fs,
        method=cfg["f0_method"],
        lowpass_hz=cfg["f0_lowpass_hz"],
        percentile=cfg["f0_percentile"],
        window_sec=cfg["f0_window_sec"],
    )

    dff = delta_f / f0
    if cfg["dff_units"] == "percent":
        dff = dff * 100.0

    z = zscore(
        dff,
        baseline_mask=baseline_mask if cfg["zscore_mode"] == "baseline" else None,
        robust=(cfg["zscore_mode"] == "robust"),
    )

    return {
        "fs": fs, "decimation_factor": decim,
        "sig_med": sig_med, "ctl_med": ctl_med,
        "sig_filt": sig_filt, "ctl_filt": ctl_filt,
        "ctl_fit": ctl_fit, "deltaF": delta_f,
        "F0_trace": f0, "F0": float(np.nanmedian(f0)),
        "dFF": dff, "z_dFF": z,
        "slope": slope, "intercept": intercept,
        "settings": cfg,
    }
