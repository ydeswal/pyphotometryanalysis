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
    "ppd_time_axis",
    "samples_to_seconds",
    "clock_columns",
    "digital_pulse_times",
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
             with_time=False, clock_source="sample") -> dict:
    """
    Read a pyPhotometry .ppd file into named signals.

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

    PC-timestamped files (a modified acquisition program; header carries
    "pc_timestamps": true and "bytes_per_sample"): each sample is the frame
    above FOLLOWED BY an 8-byte float64 PC timestamp (Unix seconds), so a
    sample is 16 bytes instead of 8. A reader that does not know this takes
    every timestamp for two extra frames of analog data: the duration doubles
    and every third and fourth "frame" is the bytes of a float, which plot as a
    sawtooth and wreck the 405 fit. See _pc_layout() and build_clock_model().

    clock_source
        "sample" (default): elapsed time comes from the board's sample counter
                  (k / fs), anchored to the PC clock at sample 0. The per-sample
                  PC timestamps are used to find the start time, to detect lost
                  samples, and to cross-check the clock, but not to stretch the
                  time axis. See build_clock_model() for why.
        "pc"    : elapsed time follows the per-sample PC timestamps (jitter
                  removed by a piecewise-linear fit).

    Returns a dict with analog_1, analog_2, digital_1, digital_2, sampling_rate,
    header, frame_layout, a "clock" dict (start time, time model, checks), and
    time_sec when with_time=True.
    """
    ppd_path = Path(ppd_path)
    dtype = np.dtype(dtype)
    with open(ppd_path, "rb") as f:
        header_size = int.from_bytes(f.read(2), "little")
        header = json.loads(f.read(header_size).decode("utf-8"))
    # Memory-map the data block instead of reading it into RAM. A 48 h, 130 Hz
    # PC-timestamped file is 360 MB; read() made all of it resident for the
    # whole analysis, while mapped pages are only touched as channels are
    # extracted and can be dropped by the OS afterwards. Every array returned
    # below is a copy, so nothing keeps the mapping alive.
    data_bytes = ppd_path.stat().st_size - 2 - header_size
    buf = (np.memmap(ppd_path, dtype=np.uint8, mode="r", offset=2 + header_size)
           if data_bytes > 0 else np.zeros(0, np.uint8))

    fs = float(header["sampling_rate"])
    mode = str(header.get("mode", "")).strip()
    version = _parse_version(header.get("version", "0.0.0"))
    read_warnings = []

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

    # Do NOT convert the whole stream at once: channels are sliced out of the
    # uint16 buffer and converted individually, in float32 (15-bit ADC values
    # fit in float32's 24-bit mantissa, so nothing is lost).

    # ---- decide the frame layout -------------------------------------------
    is_pulsed = "pulsed" in mode.lower()
    new_format = version >= (1, 1)
    words_per_frame = 2 * n_analog if (is_pulsed and new_format) else n_analog

    pc_layout = _pc_layout(header, buf, words_per_frame, fs)
    pc_t = None
    if pc_layout is not None:
        bps, wps, how = pc_layout
        if wps != words_per_frame:
            read_warnings.append(
                f"The header says {wps} words per sample but mode '{mode}' / version "
                f"{header.get('version')} implies {words_per_frame}. Using {wps}.")
            words_per_frame = wps
        n_samples_file = len(buf) // bps
        if len(buf) % bps:
            read_warnings.append(
                f"The file ends with {len(buf) % bps} bytes of an incomplete sample "
                "(acquisition was probably interrupted mid-write). They were dropped; "
                "every complete sample was recovered.")
        rec = np.frombuffer(buf, dtype=np.dtype({
            "names": ["w", "t"], "formats": [("<u2", (wps,)), "<f8"],
            "offsets": [0, 2 * wps], "itemsize": bps}), count=n_samples_file)
        frames = rec["w"]                 # (n, wps) strided view, no copy
        pc_t = rec["t"]                   # (n,) strided view, no copy
        layout_note = (f"PC-timestamped samples: {bps} bytes each = {wps} data "
                       f"words + an 8-byte PC timestamp ({how}).")
    else:
        if len(buf) % 2:
            buf = buf[:-1]
        raw = np.frombuffer(buf, dtype="<u2")
        n_samples_file = raw.size // words_per_frame
        if raw.size % words_per_frame:
            # Previously this "fell back" to the other layout whenever it
            # divided the data. That turns a file truncated mid-frame into a
            # silently misread one, so the declared layout is now kept and only
            # the incomplete last frame is dropped.
            read_warnings.append(
                f"The data block holds {raw.size} words, not a whole number of "
                f"{words_per_frame}-word frames; the incomplete last frame was dropped "
                "(acquisition was probably interrupted mid-write).")
        frames = raw[: n_samples_file * words_per_frame].reshape(-1, words_per_frame)
        layout_note = f"{words_per_frame} data words per sample, no PC timestamps."

    out = {
        "header": header,
        "sampling_rate": fs,
        "mode": mode,
        "version": header.get("version", ""),
        "subject_ID": header.get("subject_ID", ""),
        "n_analog_channels": n_analog,
        "volts_per_division": vpd,
        "layout_note": layout_note,
    }

    def _analog(col):
        return (frames[:, col] >> 1).astype(dtype)

    def _digital(col):
        return (frames[:, col] & 1).astype(np.uint8)

    adc_max = float(header.get("ADC_max_value", 32768))
    if words_per_frame == 2 * n_analog:
        # New-format pulsed: LED-on and LED-off stored separately.
        out["frame_layout"] = "pulsed_v1.1_led_on_off"
        for ch in range(n_analog):
            on = _analog(2 * ch)
            off = _analog(2 * ch + 1)
            on *= vpd[ch]
            off *= vpd[ch]
            # Same clipping rule as the official pyPhotometry importer.
            clip_thr = 0.98 * adc_max * vpd[ch]
            out[f"analog_{ch + 1}_clipping_fraction"] = float(
                np.mean(np.maximum(on, off) > clip_thr)) if on.size else 0.0
            if keep_components:
                out[f"analog_{ch + 1}_raw_LED_on"] = on.copy()
                out[f"analog_{ch + 1}_raw_baseline"] = off.copy()
            on -= off          # in place; avoids a third full-length array
            out[f"analog_{ch + 1}"] = on
            out[f"digital_{ch + 1}"] = _digital(2 * ch)
            del off
    elif words_per_frame == n_analog:
        # Continuous, or pulsed written before v1.1 (already dark-subtracted).
        out["frame_layout"] = "one_word_per_channel"
        for ch in range(n_analog):
            sig = _analog(ch)
            sig *= vpd[ch]
            out[f"analog_{ch + 1}"] = sig
            out[f"digital_{ch + 1}"] = _digital(ch)
    else:
        raise ValueError(
            f"{ppd_path.name}: {words_per_frame} words per sample does not match "
            f"{n_analog} analog channels in either known layout.")

    n_samples = len(out["analog_1"])
    clock = build_clock_model(header, fs, n_samples, pc_t, clock_source=clock_source)
    if keep_components and pc_t is not None:
        out["pc_time_unix"] = np.array(pc_t, dtype=np.float64)
    del frames, pc_t
    if pc_layout is not None:
        del rec
    else:
        del raw
    del buf

    clock["warnings"] = read_warnings + clock["warnings"]
    out["clock"] = clock
    out["n_samples"] = n_samples
    out["duration_sec"] = clock["duration_sec"]
    # The time axis is NOT materialised by default (180 MB of float64 at 48 h,
    # 130 Hz). Callers that need it ask for it, or use ppd_time_axis().
    if with_time:
        out["time_sec"] = ppd_time_axis(clock, n_samples)
    return out


# =============================================================================
# PC CLOCK
# =============================================================================

_UNIX_LO, _UNIX_HI = 946_684_800.0, 4_102_444_800.0     # years 2000 .. 2100


def _ts_view(buf, wps, bps, count):
    return np.frombuffer(buf, dtype=np.dtype({
        "names": ["t"], "formats": ["<f8"], "offsets": [2 * wps],
        "itemsize": bps}), count=count)["t"]


def _plausible_timestamps(t, fs):
    """Do these look like per-sample Unix timestamps at roughly 1/fs spacing?"""
    t = np.asarray(t, dtype=float)
    if t.size < 3 or not np.all(np.isfinite(t)):
        return False
    if t.min() < _UNIX_LO or t.max() > _UNIX_HI:
        return False
    d = np.diff(t)
    if np.mean(d >= -1e-3) < 0.99:
        return False
    med = float(np.median(d))
    return 0.5 / fs < med < 2.0 / fs


def _pc_layout(header, buf, words_per_frame, fs):
    """
    Work out whether every sample is followed by an 8-byte PC timestamp.

    Returns (bytes_per_sample, words_per_sample, how) or None for a standard
    file. The header is trusted first ("pc_timestamps", "bytes_per_sample",
    "words_per_sample"). If it says nothing, the data are sniffed: a standard
    file can only be mistaken for a timestamped one if its analog words happen
    to spell out a monotonic run of float64 Unix times ticking at 1/fs, which
    does not happen by chance. This catches timestamped files written by an
    earlier version of the modified acquisition code that did not label itself.
    """
    wps_h = header.get("words_per_sample")
    bps_h = header.get("bytes_per_sample")
    declared = bool(header.get("pc_timestamps")) or (
        bps_h is not None and int(bps_h) != 2 * int(wps_h or words_per_frame))
    if declared:
        wps = int(wps_h or words_per_frame)
        bps = int(bps_h or 2 * wps + 8)
        if bps - 2 * wps != 8:
            raise ValueError(
                f"Header declares {bps} bytes per sample for {wps} data words; only "
                "an 8-byte (float64) PC timestamp per sample is supported.")
        fmt = str((header.get("pc_clock") or {}).get("format", "float64"))
        if "float64" not in fmt and "f8" not in fmt:
            raise ValueError(f"Unsupported PC timestamp format '{fmt}'.")
        n = len(buf) // bps
        if n >= 3 and not _plausible_timestamps(_ts_view(buf, wps, bps, min(n, 2000)), fs):
            warnings.warn("Header declares PC timestamps but the first ones do not "
                          "look like Unix seconds at the sampling rate.")
        return bps, wps, "declared in the header"

    for wps in dict.fromkeys((words_per_frame, 4, 2)):
        bps = 2 * wps + 8
        n = len(buf) // bps
        if n < 3:
            continue
        if _plausible_timestamps(_ts_view(buf, wps, bps, min(n, 2000)), fs):
            return bps, wps, "detected from the data; the header does not declare it"
    return None


def _parse_local(text):
    """Header times are local wall-clock ISO strings, possibly with an offset."""
    if not text:
        return None
    try:
        ts = pd.Timestamp(str(text))
    except (ValueError, TypeError):
        return None
    return ts.tz_localize(None) if ts.tzinfo is not None else ts


def _block_levels(t_pc, fs, block):
    """
    Per block of samples: (centre index, 5th-percentile of PC time minus the
    nominal sample time). A timestamp is the moment the PC RECEIVED the sample,
    i.e. acquisition time plus a USB/OS delay that is never negative, so the
    lower envelope is the best estimate of when samples were actually taken.
    """
    n = t_pc.size
    t_ref = float(t_pc[0]) if np.isfinite(t_pc[0]) else float(np.nanmin(t_pc[:1000]))
    centres, levels = [], []
    for s in range(0, n, block):
        e = min(n, s + block)
        off = np.asarray(t_pc[s:e], dtype=np.float64) - t_ref - np.arange(s, e) / fs
        off = off[np.isfinite(off)]
        if off.size < max(5, (e - s) // 4):
            continue
        centres.append((s + e - 1) / 2.0)
        levels.append(float(np.percentile(off, 5)))
    return t_ref, np.asarray(centres), np.asarray(levels)


def _find_steps(centres, levels, thr):
    """Persistent jumps in the block levels: (block index, size in seconds)."""
    steps, j, nb = [], 0, levels.size
    d = np.diff(levels)
    while j < nb - 1:
        if abs(d[j]) > thr:
            before = float(np.median(levels[max(0, j - 2): j + 1]))
            after = float(np.median(levels[j + 1: min(nb, j + 4)]))
            if abs(after - before) > thr:
                steps.append((j, after - before))
                j += 3
                continue
        j += 1
    return steps


def _refine_step(t_pc, t_ref, fs, lo, hi, before, step):
    """Sample index where a step of `step` seconds happens inside [lo, hi)."""
    k = np.arange(lo, hi)
    off = np.asarray(t_pc[lo:hi], dtype=np.float64) - t_ref - k / fs
    off = np.where(np.isfinite(off), off, np.nan)
    half = before + step / 2.0
    if step > 0:
        # Arrival delays only push timestamps later, so the first sample from
        # which every later timestamp stays above the midpoint is the jump.
        rev_min = np.fmin.accumulate(off[::-1])[::-1]
        idx = np.flatnonzero(rev_min > half)
    else:
        idx = np.flatnonzero(off < half)
    return int(lo + (idx[0] if idx.size else (hi - lo) // 2))


def build_clock_model(header, fs, n_samples, pc_t=None, clock_source="sample"):
    """
    Map sample index -> elapsed seconds and wall-clock time, and check it.

    Why the default time axis is the board's sample clock, not the raw PC stamps
    ---------------------------------------------------------------------------
    A PC timestamp is when the PC RECEIVED a sample over USB, not when it was
    taken: individual stamps jitter by milliseconds. More importantly, the
    modified acquisition code derives them from the Windows performance
    counter, anchored to the wall clock once at the start. The performance
    counter and the Windows wall clock (the clock FED3 PC_Timestamp, LIQ HD
    pc_clock and anything else using the system time are on) can run at
    slightly different rates. On m10_fasted-2026-10-14 the per-sample stamps
    gained 2.4 s on the sample counter over 23.9 h, while the header's end_time,
    written from the wall clock when recording stopped, sits 0.3 s after the
    sample-clock prediction and 2.1 s BEFORE the last PC stamp, which is only
    possible if the stamps drifted. So the sample counter, anchored once to the
    wall clock, tracks wall-clock time better than the stamps do. The stamps
    remain essential for three things: the start time, detecting samples that
    were lost (a step in PC time the sample counter does not show), and this
    cross-check. clock_source="pc" follows the stamps instead.

    Returns a dict:
      source          "sample" | "pc" | "header" (no PC stamps; start from
                      the header date_time)
      start_unix      Unix time of sample 0 (None without PC stamps)
      start_local     local wall-clock time of sample 0, ISO string
      utc_offset_s    from the header, when recorded
      segments        list of {i0, i1, t0, dt}: elapsed(i) = t0 + (i - i0) * dt
      duration_sec, pc_drift_ppm, pc_drift_sec, gaps, end_time_check,
      notes, warnings
    """
    notes, warns = [], []
    pc_info = header.get("pc_clock") or {}
    utc_offset = pc_info.get("utc_offset_s")
    clock = {
        "source": "header", "has_pc_timestamps": pc_t is not None,
        "start_unix": None, "start_local": None, "utc_offset_s": utc_offset,
        "segments": [{"i0": 0, "i1": int(n_samples), "t0": 0.0, "dt": 1.0 / fs}],
        "duration_sec": n_samples / fs, "gaps": [], "pc_drift_ppm": None,
        "pc_drift_sec": None, "pc_jitter_ms": None, "end_time_check": None,
        "notes": notes, "warnings": warns,
    }
    start_hdr = _parse_local(header.get("date_time"))

    if pc_t is None and header.get("first_sample_unix") is not None:
        # A file written by convert_ppd.py: the PC timestamps were moved to a
        # sidecar, but the PC-clock start time and any lost-sample gaps were
        # kept in the header.
        clock.update(source="sample", start_unix=float(header["first_sample_unix"]),
                     start_local=header.get("first_sample_local"),
                     utc_offset_s=header.get("utc_offset_s", utc_offset),
                     pc_drift_ppm=header.get("pc_clock_drift_ppm"))
        segs, total = [], 0.0
        gaps = sorted(header.get("lost_sample_gaps") or [], key=lambda g: g["at_sample"])
        bounds = [0] + [int(g["at_sample"]) for g in gaps] + [int(n_samples)]
        for s in range(len(bounds) - 1):
            if s > 0:
                total += float(gaps[s - 1]["gap_sec"])
            segs.append({"i0": bounds[s], "i1": bounds[s + 1],
                         "t0": bounds[s] / fs + total, "dt": 1.0 / fs})
        clock["segments"] = segs
        clock["gaps"] = gaps
        clock["duration_sec"] = segs[-1]["t0"] + (segs[-1]["i1"] - segs[-1]["i0"]) / fs
        notes.append(
            f"Converted from a PC-timestamped file ({header.get('converted_from_pc_timestamped_ppd')}). "
            f"First sample at {str(clock['start_local'])[:23].replace('T', ' ')} on the PC clock"
            + (f"; {len(gaps)} lost-sample gap(s) restored from the header." if gaps else "."))
        _end_time_check(clock, header, start_hdr, fs, n_samples)
        return clock

    if pc_t is None or n_samples < 10:
        if start_hdr is not None:
            clock["start_local"] = start_hdr.isoformat()
            notes.append(
                f"No per-sample PC timestamps. Clock times are the header start time "
                f"({start_hdr:%Y-%m-%d %H:%M:%S}) plus elapsed time on the sample clock.")
        _end_time_check(clock, header, start_hdr, fs, n_samples)
        return clock

    finite = np.isfinite(pc_t)
    if finite.mean() < 0.99 or not _plausible_timestamps(pc_t[:2000][finite[:2000]], fs):
        warns.append("The per-sample PC timestamps are missing or implausible; they "
                     "were ignored and the sample clock was used.")
        clock["has_pc_timestamps"] = False
        if start_hdr is not None:
            clock["start_local"] = start_hdr.isoformat()
        return clock

    block = max(int(round(30.0 * fs)), 50)          # 30-second blocks
    t_ref, centres, levels = _block_levels(pc_t, fs, block)
    thr = max(0.2, 3.0 / fs)
    steps = _find_steps(centres, levels, thr)

    # Split into segments at persistent steps. A forward step means the PC saw
    # time pass that the sample counter did not: samples were lost (or the PC
    # clock was moved forward). A backward step cannot be lost samples; it is
    # the PC clock being set back, and is ignored for the sample-clock axis.
    bounds, gaps, ignored = [0], [], []
    for j, size in steps:
        lo = int(max(0, centres[j] - block))
        hi = int(min(n_samples, centres[min(j + 1, centres.size - 1)] + block))
        before = float(np.median(levels[max(0, j - 2): j + 1]))
        i_step = _refine_step(pc_t, t_ref, fs, lo, hi, before, size)
        if size > 0:
            bounds.append(i_step)
            gaps.append({"at_sample": i_step, "gap_sec": float(size)})
        else:
            ignored.append((i_step, float(size)))
    bounds.append(int(n_samples))

    # Linear fit of PC time against sample index within each segment, on the
    # block lower envelopes (jitter-free). level(c) = a + b*c.
    seg_fits = []
    for s in range(len(bounds) - 1):
        i0, i1 = bounds[s], bounds[s + 1]
        sel = (centres >= i0) & (centres < i1)
        if sel.sum() >= 2:
            b, a = np.polyfit(centres[sel], levels[sel], 1)
        elif sel.sum() == 1:
            a, b = float(levels[sel][0]), 0.0
        else:
            k = np.arange(i0, min(i1, i0 + block))
            o = np.asarray(pc_t[i0:i0 + k.size], float) - t_ref - k / fs
            a, b = float(np.nanpercentile(o, 5)), 0.0
        seg_fits.append((i0, i1, float(a), float(b)))

    start_unix = t_ref + seg_fits[0][2]               # fitted PC time of sample 0
    clock["start_unix"] = float(start_unix)
    tz = None
    if utc_offset is not None:
        import datetime as _dt
        tz = _dt.timezone(_dt.timedelta(seconds=float(utc_offset)))
        clock["start_local"] = pd.Timestamp(start_unix, unit="s", tz="UTC") \
            .tz_convert(tz).tz_localize(None).isoformat()
    elif start_hdr is not None:
        # No offset recorded: take it from the header's local start time.
        off = round((start_hdr - pd.Timestamp(start_unix, unit="s")).total_seconds() / 900) * 900
        clock["utc_offset_s"] = utc_offset = int(off)
        clock["start_local"] = (pd.Timestamp(start_unix, unit="s")
                                + pd.Timedelta(seconds=off)).isoformat()

    # Diagnostics on the longest segment.
    longest = max(seg_fits, key=lambda f: f[1] - f[0])
    clock["pc_drift_ppm"] = float(longest[3] * fs * 1e6)
    span = n_samples / fs
    clock["pc_drift_sec"] = float(longest[3] * fs * span)
    sl = slice(0, min(n_samples, 20 * block))
    k = np.arange(sl.stop)
    resid = (np.asarray(pc_t[sl], float) - t_ref - k / fs
             - (seg_fits[0][2] + seg_fits[0][3] * k))
    clock["pc_jitter_ms"] = float(np.nanstd(resid) * 1e3)

    # Build the time model.
    source = clock_source if clock_source in ("sample", "pc") else "sample"
    if source == "pc" and ignored:
        warns.append("The PC clock jumped backwards during the recording, so the "
                     "PC-timestamp time axis would run backwards; the sample clock "
                     "was used instead.")
        source = "sample"
    segs, gap_total = [], 0.0
    for s, (i0, i1, a, b) in enumerate(seg_fits):
        if source == "pc":
            t0 = (i0 / fs + a + b * i0) - seg_fits[0][2]
            dt = 1.0 / fs + b
        else:
            if s > 0:
                gap_total += gaps[s - 1]["gap_sec"]
            t0 = i0 / fs + gap_total
            dt = 1.0 / fs
        segs.append({"i0": int(i0), "i1": int(i1), "t0": float(t0), "dt": float(dt)})
    clock["segments"] = segs
    clock["source"] = source
    last = segs[-1]
    clock["duration_sec"] = float(last["t0"] + (last["i1"] - last["i0"]) * last["dt"])
    for g, sg in zip(gaps, segs[1:]):
        g["at_elapsed_sec"] = sg["t0"] - g["gap_sec"]
    clock["gaps"] = gaps

    start_txt = (pd.Timestamp(clock["start_local"]).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                 if clock["start_local"] else f"Unix {start_unix:.3f}")
    off_txt = (f" (UTC{int(utc_offset) // 3600:+03d}:{abs(int(utc_offset)) % 3600 // 60:02d})"
               if utc_offset is not None else "")
    notes.append(
        f"First sample taken at {start_txt}{off_txt} on the PC clock. "
        f"PC timestamp jitter {clock['pc_jitter_ms']:.1f} ms (USB delivery).")
    notes.append(
        f"PC timestamps vs the board's sample clock: {clock['pc_drift_ppm']:+.1f} ppm, "
        f"i.e. {clock['pc_drift_sec']:+.2f} s over the {span / 3600:.2f} h recording.")
    notes.append(
        "Time axis: " + ("board sample clock, anchored to the PC clock at the first "
                         "sample (default)." if source == "sample" else
                         "per-sample PC timestamps (piecewise-linear fit)."))
    if not gaps and not ignored:
        notes.append("No lost samples: the PC timestamps run continuously from the "
                     "first sample to the last.")
    for g in gaps:
        warns.append(
            f"The PC timestamps jump forward by {g['gap_sec']:.2f} s at elapsed "
            f"{g['at_elapsed_sec'] / 3600:.3f} h (sample {g['at_sample']:,}) while the "
            "sample counter does not, so samples were lost there (or the PC clock was "
            "moved forward). The time axis includes the gap; filtering still treats "
            "the data on either side as continuous.")
    for i_step, size in ignored:
        notes.append(f"The PC clock was set back by {-size:.2f} s near sample {i_step:,}. "
                     "That cannot be lost samples and does not affect the time axis.")
    if abs(clock["pc_drift_ppm"]) > 200:
        warns.append(f"The PC timestamps and the sample clock disagree by "
                     f"{clock['pc_drift_ppm']:.0f} ppm, far more than two crystals "
                     "should. Check the sampling rate in the header.")

    _end_time_check(clock, header, start_hdr, fs, n_samples)
    return clock


def _end_time_check(clock, header, start_hdr, fs, n_samples):
    """Compare the header's end_time (wall clock at stop) with both predictions."""
    end_hdr = _parse_local(header.get("end_time"))
    start = _parse_local(clock.get("start_local")) or start_hdr
    if end_hdr is None or start is None:
        return
    segs = clock["segments"]
    last_elapsed = segs[-1]["t0"] + (segs[-1]["i1"] - 1 - segs[-1]["i0"]) * segs[-1]["dt"]
    pred = start + pd.Timedelta(seconds=last_elapsed)
    lag = (end_hdr - pred).total_seconds()
    chk = {"end_time": end_hdr.isoformat(), "predicted_last_sample": pred.isoformat(),
           "end_minus_last_sample_sec": float(lag)}
    if clock.get("pc_drift_sec") is not None and clock["source"] == "sample":
        pc_pred = pred + pd.Timedelta(seconds=clock["pc_drift_sec"])
        chk["pc_stamp_last_sample"] = pc_pred.isoformat()
        chk["end_minus_pc_last_sample_sec"] = float((end_hdr - pc_pred).total_seconds())
    clock["end_time_check"] = chk
    if -0.5 <= lag <= 10:
        clock["notes"].append(
            f"Cross-check: the header end_time ({end_hdr.strftime('%H:%M:%S.%f')[:-3]}) "
            f"falls {lag:.2f} s after the last sample on this time axis, as it should.")
        if chk.get("end_minus_pc_last_sample_sec", 0) < -0.5:
            clock["notes"].append(
                f"The raw PC stamps would put the last sample "
                f"{-chk['end_minus_pc_last_sample_sec']:.2f} s AFTER end_time, which is "
                "impossible, so they drifted from the wall clock; that is why the time "
                "axis follows the sample clock.")
    else:
        msg = (f"The header end_time ({end_hdr.strftime('%H:%M:%S.%f')[:-3]}) is "
               f"{lag:+.2f} s from the last sample on this time axis "
               f"({pred.strftime('%H:%M:%S.%f')[:-3]}); it should be slightly after it.")
        if clock["source"] == "pc":
            msg += (" The per-sample PC timestamps have drifted from the wall clock. For "
                    "aligning with other devices that log the PC wall clock (FED3 "
                    "PC_Timestamp, LIQ HD pc_clock), use the sample-clock time axis.")
        else:
            msg += " Check the time axis before aligning to other devices."
        clock["warnings"].append(msg)


def samples_to_seconds(clock, idx):
    """Elapsed seconds for (possibly fractional) sample indices."""
    idx = np.asarray(idx, dtype=np.float64)
    segs = clock["segments"]
    if len(segs) == 1:
        s = segs[0]
        return s["t0"] + (idx - s["i0"]) * s["dt"]
    starts = np.array([s["i0"] for s in segs], dtype=np.float64)
    which = np.clip(np.searchsorted(starts, idx, side="right") - 1, 0, len(segs) - 1)
    t0 = np.array([s["t0"] for s in segs])[which]
    i0 = starts[which]
    dt = np.array([s["dt"] for s in segs])[which]
    return t0 + (idx - i0) * dt


def ppd_time_axis(clock, n_samples, step=1):
    """Elapsed seconds of samples 0, step, 2*step, ... (float64)."""
    n_out = int(n_samples)
    return samples_to_seconds(clock, np.arange(n_out, dtype=np.float64) * step)


def clock_columns(clock, time_sec):
    """
    Wall-clock columns for an elapsed-time vector:
      clock_unix  Unix seconds (only when the start is known in absolute time)
      clock_time  local wall-clock time, 'YYYY-MM-DD HH:MM:SS.fff'
    Returns {} when the recording carries no start time.
    """
    time_sec = np.asarray(time_sec, dtype=np.float64)
    cols = {}
    if clock.get("start_unix") is not None:
        cols["clock_unix"] = clock["start_unix"] + time_sec
    if clock.get("start_local"):
        start = pd.Timestamp(clock["start_local"])
        ts = start + pd.to_timedelta(time_sec, unit="s")
        cols["clock_time"] = ts.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]
    return cols


def digital_pulse_times(digital: np.ndarray, fs: float, clock=None) -> np.ndarray:
    """Rising-edge times (seconds) of a digital channel. Used for lickometer TTL.

    With a clock model the times follow the recording's time axis (including
    any gaps); without one they are index / fs."""
    d = np.asarray(digital).astype(np.int8)
    if d.size < 2:
        return np.array([], dtype=float)
    rising = np.flatnonzero(np.diff(d) == 1) + 1
    if clock is not None:
        return samples_to_seconds(clock, rising)
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
