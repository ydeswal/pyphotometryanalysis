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
* pyPhotometry .ppd binary layout follows the official spec and the official
  importer (pyPhotometry tools/data_import.py, Akam & Walton 2019 Sci Rep 9:3521).
* Default pipeline ("douglass2025") follows the STAR Methods of
  Douglass AM ... Lowell BB (2025) Cell Metabolism 37:708-722,
  doi:10.1016/j.cmet.2024.11.009:
    - median filter, then 10 Hz low-pass;
    - motion correction = linear best fit of 405 onto 465, subtracted from 465;
    - dF/F0 = (465 - fitted 405) / F0, F0 = 0.001 Hz low-pass of 465 (whole session);
    - 30-minute running average for long-term traces; session z-scores.
  Details the paper does not state (median kernel, filter order) are taken
  from the pyPhotometry reference implementation the paper cites (ref. 30).
* Alternative pipeline ("pyphotometry") reproduces pyPhotometry's
  preprocess_data: double-exponential bleaching correction, regression on the
  bleach-corrected traces, dF/F = motion-corrected signal / exponential fit.
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
    # Which published pipeline to run.
    #   "douglass2025" : Douglass et al. 2025 Cell Metab (default)
    #   "pyphotometry" : Akam & Walton pyPhotometry preprocess_data
    "pipeline": "douglass2025",
    # Median filter. The paper says "median-filtered" without a kernel size;
    # pyPhotometry specifies the kernel in SAMPLES, so that is the default here.
    "median_kernel_samples": 5,
    "median_filter_sec": 0.0,      # if > 0, overrides the sample kernel
    "lowpass_hz": 10.0,            # paper: 10 Hz cutoff
    "filter_order": 2,             # pyPhotometry uses butter(2, ...)
    "fit_method": "ols",           # paper: "linear best fit" (= OLS / linregress)
    "irls_tuning": 1.345,          # only used when fit_method == "irls"
    # A motion artefact moves 405 and 465 in the SAME direction, so the slope of
    # 465-on-405 must be positive. A negative slope means the fit is tracking
    # biology (or 405 is not truly isosbestic), not motion.
    #   "zero"  : clamp the slope to 0 (no motion subtraction) and warn
    #   "allow" : keep whatever the regression returns (paper-literal)
    "negative_slope_policy": "zero",
    "f0_method": "lowpass",        # "lowpass" | "rolling_percentile" | "percentile" | "median"
    "f0_lowpass_hz": 0.001,        # paper: F0 = 0.001 Hz low-pass of 465
    "f0_percentile": 10.0,
    "f0_window_sec": 1800.0,
    "dff_units": "ratio",          # internal unit; the app displays x100 (%)
    "zscore_mode": "session",      # paper: session z-score
    "long_term_smooth_sec": 1800.0,  # paper: 30-min running average
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
            # Same clipping rule as the official pyPhotometry importer.
            adc_max = float(header.get("ADC_max_value", 32768))
            clip_thr = 0.98 * adc_max * vpd[ch]
            out[f"analog_{ch + 1}_clipping_fraction"] = float(
                np.mean(np.maximum(on, off) > clip_thr)) if n else 0.0
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

def median_filter_trace(values: np.ndarray, fs: float, window_sec: float = 0.0,
                        kernel_samples: int = 0) -> np.ndarray:
    """
    Median filter with edge reflection.

    kernel_samples : odd kernel width in samples (pyPhotometry convention).
    window_sec     : alternative width in seconds; used instead when > 0.

    Uses scipy.ndimage.median_filter (rolling histogram, reflected edges)
    rather than scipy.signal.medfilt, which is O(n*k) and zero-pads the ends.
    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    if window_sec is not None and window_sec > 0:
        k = int(round(float(window_sec) * float(fs)))
    else:
        k = int(kernel_samples or 0)
    if k < 3:
        return values.copy()
    if k % 2 == 0:
        k += 1
    if k >= values.size:
        return np.full_like(values, float(np.nanmedian(values)))
    return _nd_median_filter(values, size=k, mode="reflect")


def _block_mean(values, step):
    """Average non-overlapping blocks of `step` samples (an anti-alias filter).

    Returns (block_means, block_centre_indices)."""
    values = np.asarray(values)
    n = (values.size // step) * step
    head = values[:n].astype(np.float64).reshape(-1, step).mean(axis=1)
    centres = np.arange(head.size, dtype=float) * step + (step - 1) / 2.0
    if n < values.size:
        head = np.append(head, float(np.mean(values[n:])))
        centres = np.append(centres, n + (values.size - n - 1) / 2.0)
    return head, centres


def lowpass(values, fs, cutoff_hz, order=2):
    """
    Zero-phase Butterworth low-pass (forward + backward pass, as scipy filtfilt).

    Edge handling
    -------------
    A zero-phase filter needs data beyond each end of the trace. scipy's default
    pads only 3*(2*order) samples. For the 0.001 Hz F0 filter that is a fraction
    of a second against a ~16-minute time constant, so the filter just holds the
    value the trace STARTS at: on a short recording "F0" became the first sample
    rather than a baseline. Here the trace is mirror-padded by three filter
    periods (3/cutoff seconds), so the output converges to what an ideal
    low-pass returns: the slow component, which for a recording shorter than
    the filter period is the session mean.

    Very low cutoffs
    ----------------
    Below a normalised cutoff of 1e-3 the trace is block-averaged down to ~20x
    the cutoff, filtered, and interpolated back. Block averaging is itself a
    low-pass filter, so, unlike the previous point-sampling (values[::step]),
    fast fluctuations are not aliased into the baseline.

    Cutoff at or above Nyquist
    --------------------------
    A signal sampled at fs contains nothing above fs/2, so the filter would be
    the identity; the trace is returned unchanged. At 20 Hz sampling the
    paper's 10 Hz low-pass is exactly this case.
    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        values = values.astype(np.float32)
    in_dtype = values.dtype
    fs = float(fs)
    if cutoff_hz is None or cutoff_hz <= 0 or values.size < 12:
        return values.copy()

    nyquist = fs / 2.0
    if float(cutoff_hz) >= 0.99 * nyquist:
        return values.copy()
    cutoff = float(cutoff_hz)
    wn = cutoff / nyquist

    if wn < 1e-3:
        target_fs = max(cutoff * 20.0, 1e-6)
        step = max(int(np.floor(fs / target_fs)), 1)
        # Keep enough blocks for the filter to operate on; a short recording
        # would otherwise collapse to a handful of points that are returned
        # unfiltered.
        step = min(step, max(1, values.size // 64))
        if step > 1:
            small, centres = _block_mean(values, step)
            filt_small = lowpass(small, fs / step, cutoff, order=order)
            return np.interp(np.arange(values.size, dtype=float), centres,
                             filt_small).astype(in_dtype, copy=False)

    sos = butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    pad = int(np.ceil(3.0 * fs / cutoff))
    pad = max(pad, 3 * (2 * order) + 1)

    # Long traces are filtered in overlapping blocks so peak memory does not
    # scale with recording length. Each block keeps a margin on both sides and
    # only its interior is used, so the result equals filtering in one go.
    if values.size > _FILTER_BLOCK:
        margin = min(max(pad, int(round(20.0 * fs / cutoff))), values.size // 2)
        padded = np.pad(values, margin, mode="symmetric")
        out = np.empty_like(values)
        for start in range(0, values.size, _FILTER_BLOCK):
            stop = min(start + _FILTER_BLOCK, values.size)
            seg = sosfiltfilt(sos, padded[start: stop + 2 * margin].astype(np.float64),
                              padtype=None)
            out[start:stop] = seg[margin: margin + (stop - start)].astype(in_dtype, copy=False)
            del seg
        return out

    pad = min(pad, 5_000_000)
    padded = np.pad(values.astype(np.float64), pad, mode="symmetric")
    filt = sosfiltfilt(sos, padded, padtype=None)
    return filt[pad: pad + values.size].astype(in_dtype, copy=False)


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


def fit_control_to_signal(control, signal, method="ols", tuning=1.345,
                          negative_slope_policy="zero"):
    """
    Scale the 405 control onto the 465 signal:

        fitted_405(t) = a * 405(t) + b
        (a, b) = argmin  sum_t [ 465(t) - a*405(t) - b ]^2      (OLS)

    Closed form:  a = cov(405, 465) / var(405),   b = mean(465) - a*mean(405).

    Returns (fitted_control, slope, intercept, info). info holds the
    unconstrained slope, R^2 of the fit, the 405-465 Pearson r, and whether the
    slope was clamped.

    Why a negative slope is clamped
    -------------------------------
    The 405 channel exists because movement, fibre bending and haemodynamics
    change the light collected at BOTH wavelengths in the same direction. Shared
    artefacts can only produce a POSITIVE slope. A negative slope appears when
    465 contains a large real change and 405 drifts slightly the other way,
    which happens because GCaMP fluorescence excited at 405 nm still falls a
    little as calcium rises (the true isosbestic point is a few nm higher). The
    regression then multiplies the small, noisy 405 trace by a large negative
    number to "explain" the real signal, and subtracting it deletes the biology
    and injects amplified 405 noise. Policy "zero" applies the constraint
    a >= 0; the constrained least-squares solution is then a = 0, b = mean(465).
    """
    control = np.asarray(control, dtype=float)
    signal = np.asarray(signal, dtype=float)
    good = np.isfinite(control) & np.isfinite(signal)
    if good.sum() < 3:
        raise ValueError("Not enough finite samples to fit 405 onto 465.")

    x, y = control[good], signal[good]
    if method == "irls":
        slope, intercept = _irls_fit(x, y, tuning=tuning)
    else:
        slope, intercept = _ols_fit(x, y)

    raw_slope, raw_intercept = slope, intercept
    clamped = False
    if slope < 0 and negative_slope_policy == "zero":
        slope, intercept = 0.0, float(np.mean(y))
        clamped = True

    fitted = slope * control + intercept
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    r_xy = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
    info = {"raw_slope": float(raw_slope), "raw_intercept": float(raw_intercept),
            "slope_clamped": clamped, "r2": r2, "pearson_r_405_465": r_xy}
    return fitted, float(slope), float(intercept), info


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


def _double_exponential(t, const, amp_fast, amp_slow, tau_fast, tau_slow):
    return const + amp_slow * np.exp(-t / tau_slow) + amp_fast * np.exp(-t / tau_fast)


def fit_double_exponential(values, fs):
    """
    pyPhotometry photobleaching model

        B(t) = c + A_fast * exp(-t/tau_fast) + A_slow * exp(-t/tau_slow)

    fitted on a 1 Hz block-averaged copy with the same start values and bounds
    as pyPhotometry tools/data_import.py (_fit_exponential). Returns the curve
    at full rate, or None if the fit does not converge.
    """
    from scipy.optimize import curve_fit
    values = np.asarray(values, dtype=float)
    step = max(int(round(fs)), 1)
    small, centres = _block_mean(values, step)
    t_small = centres / fs
    max_sig = float(np.max(values))
    if not np.isfinite(max_sig) or max_sig <= 0 or small.size < 6:
        return None
    p0 = [max_sig / 2, max_sig / 4, max_sig / 4, 300, 3000]
    bounds = ([0, 0, 0, 60, 600], [max_sig, max_sig, max_sig, 600, 36000])
    try:
        params, _ = curve_fit(_double_exponential, t_small, small, p0=p0,
                              bounds=bounds, maxfev=10000)
    except Exception:
        return None
    return _double_exponential(np.arange(values.size) / fs, *params)


def process_photometry(signal_raw, control_raw, fs, settings=None,
                       baseline_mask=None, analysis_hz=None, consume=False):
    """
    Run the full pipeline on one recording.

    signal_raw  : 465 nm channel (LED-on minus LED-off, volts)
    control_raw : 405 nm channel (LED-on minus LED-off, volts)

    pipeline = "douglass2025" (default; Douglass et al. 2025 STAR Methods)
      1. median filter both channels
      2. 10 Hz zero-phase low-pass both channels
      3. fitted405 = a*405 + b, (a, b) by least squares of 465 on 405
      4. dF   = 465 - fitted405
      5. F0   = 0.001 Hz low-pass of 465 over the whole session
      6. dF/F0 = dF / F0
      7. z    = (dF/F0 - mean) / SD over the session

    pipeline = "pyphotometry" (Akam & Walton preprocess_data)
      1-2 as above
      3. B465, B405 = double-exponential bleaching fits
      4. 465bc = 465 - B465 ; 405bc = 405 - B405
      5. motion = a*405bc + b  (least squares of 465bc on 405bc)
      6. dF = 465bc - motion ; F0 = B465 ; dF/F0 = dF / F0
      7. z as above
    """
    cfg = dict(DEFAULTS)
    if settings:
        cfg.update({k: v for k, v in settings.items() if v is not None})

    signal_raw = np.asarray(signal_raw)
    control_raw = np.asarray(control_raw)
    keep = cfg.get("keep_intermediates", False)
    warnings_out = []
    notes_out = []
    fs_in = float(fs)

    def _prep(raw):
        med = median_filter_trace(raw, fs, cfg["median_filter_sec"],
                                  cfg.get("median_kernel_samples", 0))
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

    if cfg["lowpass_hz"] >= 0.99 * fs_in / 2.0:
        notes_out.append(
            f"The {cfg['lowpass_hz']:g} Hz low-pass is at or above the Nyquist frequency "
            f"({fs_in / 2:g} Hz) for {fs_in:g} Hz sampling, so it has nothing to remove "
            "and was skipped. This is expected at 20 Hz and matches the paper.")

    # Optional decimation, applied only after low-pass filtering.
    decim = 1
    if analysis_hz and analysis_hz > 0 and fs > analysis_hz:
        target = max(float(analysis_hz), 2.5 * cfg["lowpass_hz"])
        decim = int(max(1, np.floor(fs / target)))
        if decim > 1:
            sig_filt = decimate_after_lowpass(sig_filt, decim)
            ctl_filt = decimate_after_lowpass(ctl_filt, decim)
            fs = fs / decim

    duration = sig_filt.size / float(fs)
    pipeline = cfg.get("pipeline", "douglass2025")
    sig_exp = ctl_exp = None

    if pipeline == "pyphotometry":
        sig_exp = fit_double_exponential(sig_filt, fs)
        ctl_exp = fit_double_exponential(ctl_filt, fs)
        if sig_exp is None or ctl_exp is None:
            warnings_out.append(
                "Double-exponential bleaching fit did not converge; fell back to the "
                "Douglass 2025 pipeline for this recording.")
            pipeline = "douglass2025"

    if pipeline == "pyphotometry":
        sig_bc = sig_filt - sig_exp
        ctl_bc = ctl_filt - ctl_exp
        motion, slope, intercept, finfo = fit_control_to_signal(
            ctl_bc, sig_bc, method=cfg["fit_method"], tuning=cfg["irls_tuning"],
            negative_slope_policy=cfg["negative_slope_policy"])
        delta_f = sig_bc - motion
        f0 = sig_exp
        ctl_fit = sig_exp + motion      # 465-scale model, for the correction plot
    else:
        ctl_fit, slope, intercept, finfo = fit_control_to_signal(
            ctl_filt, sig_filt, method=cfg["fit_method"], tuning=cfg["irls_tuning"],
            negative_slope_policy=cfg["negative_slope_policy"])
        delta_f = sig_filt - ctl_fit
        f0 = compute_f0(
            sig_filt, fs,
            method=cfg["f0_method"],
            lowpass_hz=cfg["f0_lowpass_hz"],
            percentile=cfg["f0_percentile"],
            window_sec=cfg["f0_window_sec"],
        )
        if cfg["f0_method"] == "lowpass" and duration < 1.0 / cfg["f0_lowpass_hz"]:
            warnings_out.append(
                f"Recording is {duration / 60:.1f} min long, shorter than one period of the "
                f"{cfg['f0_lowpass_hz']:g} Hz F0 filter ({1 / cfg['f0_lowpass_hz'] / 60:.1f} min). "
                "F0 is therefore essentially the session mean of the 465 signal. The paper's "
                "F0 was designed for multi-hour to multi-day recordings.")

    if finfo["slope_clamped"]:
        warnings_out.append(
            f"The 405-to-465 fit gave a NEGATIVE slope ({finfo['raw_slope']:.3g}). A motion "
            "artefact cannot do that, so the slope was set to 0 and nothing but a constant "
            "was subtracted. This usually means 465 contains a large real change that 405 "
            "does not share. Inspect the raw traces for movement artefacts by eye.")
    elif np.isfinite(finfo["r2"]) and finfo["r2"] < 0.05:
        warnings_out.append(
            f"405 explains only {100 * finfo['r2']:.1f}% of the 465 variance, so motion "
            "correction removed very little. That is fine if the animal moved little.")

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
        "fit_info": finfo,
        "pipeline_used": pipeline,
        "duration_sec": duration,
        "warnings": warnings_out,
        "notes": notes_out,
        "settings": cfg,
    }
