#!/usr/bin/env python3
"""
lickometer.py
=============

Lickometer ingestion, bout detection, peri-event photometry alignment, and
plotting.

WHAT CHANGED IN THIS VERSION

The old loader understood exactly two CSV shapes and made you tell it which one
you had and which columns to use. Anything else - a LIQ HD raw export, a
multi-bottle file, a file with clock timestamps rather than elapsed seconds, a
file of per-bin lick counts - either failed outright or, worse, loaded and gave
a wrong answer.

`load_licks_csv()` now identifies the format itself and returns lick times plus
an explicit account of what it decided and what it distrusted.

It handles:

  * LIQ HD raw exports (record_type / pc_clock / elapsed_s / channel), split
    per bottle automatically
  * one timestamp per lick, in seconds, milliseconds, minutes, or as clock
    times ("2026-09-16T13:11:15.829" or "13:11:15")
  * a time column plus one or more 0/1 state columns (one per bottle)
  * a time column plus per-bin lick COUNTS (e.g. licks per minute, or the
    hourly files LIQ HD writes)
  * files with a leading index column, extra metadata columns, or rows that are
    not licks mixed in

THE TIME-COLUMN PROBLEM, WHICH IS NOT HYPOTHETICAL

A file can contain several plausible time columns that disagree. The LIQ HD
export that prompted this rewrite has both `pc_clock` and `elapsed_s`, and
`elapsed_s` is corrupt: the clock advances 23.4 hours across the recording while
`elapsed_s` advances 82,343 hours. Picking the wrong column does not raise an
error - it silently produces a 9-year recording with a lick rate near zero.

So when a file offers more than one time base, they are cross-checked against
each other and the disagreement is reported rather than resolved silently. A
wall-clock column is trusted over a free-running counter, because a counter can
drift, wrap, or be reset by firmware while a clock cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from photometry_core import zscore

__all__ = [
    "LICK_DEFAULTS",
    "LickLoad",
    "load_licks_csv",
    "licks_from_digital",
    "licks_from_timestamp_csv",
    "licks_from_state_csv",
    "detect_bouts",
    "lick_rate_trace",
    "peri_event_matrix",
    "bout_summary_table",
    "plot_licks",
    "plot_licks_with_signal",
    "plot_peri_event",
]


LICK_DEFAULTS = {
    "min_inter_lick_sec": 0.05,    # debounce: licks closer than this are one contact
    "inter_bout_sec": 1.0,         # gap that separates two bouts
    "min_licks_per_bout": 3,       # ignore isolated stray contacts
    "rate_window_sec": 60.0,       # window for the lick-rate trace
    "pre_sec": 10.0,               # peri-event window before onset
    "post_sec": 20.0,              # peri-event window after onset
    "baseline_start_sec": -10.0,   # baseline epoch for peri-event z-scoring
    "baseline_end_sec": -2.0,
}

# Column names we recognise, lower-cased and stripped of spaces/underscores.
_CLOCK_NAMES = {"pcclock", "clock", "datetime", "date", "wallclock", "realtime",
                "timestampiso", "pctime", "computertime"}
_ELAPSED_NAMES = {"elapseds", "elapsedsec", "elapsedseconds", "elapsed",
                  "timesec", "times", "time", "t", "seconds", "sec",
                  "licktime", "licktimes", "onset", "onsetsec", "timestamp",
                  "devicems", "elapsedms", "timems", "ms", "timemin", "minutes"}
_CHANNEL_NAMES = {"channel", "ch", "bottle", "sipper", "spout", "port", "lickometer",
                  "bottleid", "channelid", "device", "cage"}
_COUNT_NAMES = {"licks", "nlicks", "lickcount", "count", "counts", "n",
                "licksperbin", "lickstotal", "totallicks"}
_STATE_NAMES = {"lick", "licking", "state", "touch", "contact", "ttl", "digital",
                "lickstate", "beam"}
# Columns that carry a number but are never a time axis. Without this a file
# like the LIQ HD export offers "peak", "dc" and "duration_ms" as candidates,
# and a metadata column can win by accident.
_NEVER_TIME = (_CHANNEL_NAMES | _STATE_NAMES | _COUNT_NAMES | {
    "durationms", "duration", "durationsec", "peak", "dc", "split", "note",
    "subject", "recordtype", "animal", "mouse", "id", "trial", "bout",
    "session", "group", "sex", "weight", "temperature", "index", "row",
})


def _norm(name) -> str:
    return "".join(str(name).lower().split()).replace("_", "").replace("-", "").replace(".", "")


# =============================================================================
# CSV AUTO-DETECTION
# =============================================================================

@dataclass
class LickLoad:
    """
    The result of reading a lick file.

    sources        {label: array of lick times in seconds from recording start}
    combined       every lick from every source, pooled and sorted
    fmt            which layout was recognised
    time_column    the column the times came from
    duration_sec   recording length, from the file itself where it says so
    start_clock    wall-clock time of t=0, when the file carries one
    notes          decisions worth knowing about
    warnings       things that look wrong in the data
    """
    sources: dict = field(default_factory=dict)
    fmt: str = "unknown"
    time_column: str = ""
    duration_sec: float = float("nan")
    start_clock: object = None
    notes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)

    @property
    def combined(self) -> np.ndarray:
        if not self.sources:
            return np.array([], dtype=float)
        return np.sort(np.concatenate([np.asarray(v, float)
                                       for v in self.sources.values()]))

    @property
    def n_licks(self) -> int:
        return int(sum(len(v) for v in self.sources.values()))

    def __repr__(self):
        return (f"LickLoad(fmt={self.fmt!r}, sources={list(self.sources)}, "
                f"n_licks={self.n_licks}, time_column={self.time_column!r})")


def _parse_clock(series):
    """
    Parse a column of wall-clock times. Returns seconds from the first entry,
    or None if it is not a clock column.
    """
    s = series.dropna()
    if s.empty:
        return None
    # Only try on strings or datetimes - a numeric column of seconds must not be
    # coerced into 1970 epoch nanoseconds and silently accepted as a clock.
    if not (pd.api.types.is_datetime64_any_dtype(s) or
            pd.api.types.is_object_dtype(s) or
            pd.api.types.is_string_dtype(s)):
        return None
    try:
        dt = pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        try:
            dt = pd.to_datetime(series, errors="coerce")
        except (ValueError, TypeError):
            return None
    if dt.notna().sum() < max(2, 0.5 * len(dt.dropna())):
        return None
    return dt


def _infer_unit(t: np.ndarray, hint: str = "") -> tuple[float, str]:
    """
    Work out whether a numeric time column is in seconds, milliseconds or
    minutes, and return the factor that converts it to seconds.

    Explicit naming wins. Otherwise the decision is made on whether the RESULTING
    RECORDING DURATION is physically plausible, not on the typical gap between
    licks. Gap-based inference looks reasonable and fails badly: a sensor that
    double-counts contacts produces gaps of a few milliseconds, which reads as a
    file in minutes, and every timestamp then gets multiplied by sixty.

    Duration is the safer signal because it has hard limits at both ends - a
    lickometer session is never three seconds and never thirty years - whereas
    inter-lick gaps vary with the animal, the debounce and the hardware.
    """
    h = _norm(hint)
    if h.endswith("ms") or "millis" in h or h == "devicems":
        return 0.001, "milliseconds (from the column name)"
    if "min" in h:
        return 60.0, "minutes (from the column name)"
    if h.endswith("s") or "sec" in h:
        return 1.0, "seconds (from the column name)"

    finite = t[np.isfinite(t)]
    if finite.size < 3:
        return 1.0, "seconds (assumed; too few events to infer)"
    span = float(np.nanmax(finite) - np.nanmin(finite))
    if span <= 0:
        return 1.0, "seconds (assumed; all timestamps identical)"

    # A plausible session: at least 30 s, at most 10 days.
    LO, HI = 30.0, 10 * 86400.0
    for factor, label in ((1.0, "seconds"), (0.001, "milliseconds"),
                          (60.0, "minutes")):
        dur = span * factor
        if LO <= dur <= HI:
            return factor, (f"{label} (a span of {span:g} gives a "
                            f"{dur/3600:.2f} h recording)")
    return 1.0, (f"seconds (assumed; a span of {span:g} gives "
                 f"{span/3600:.2f} h, which is unusual - check the units)")


def _pick_time_base(df, out: LickLoad, prefer_column=None):
    """
    Choose the time base, cross-checking every candidate the file offers.

    Columns are considered in order of how much the file itself tells us:
    a column NAMED like a clock or an elapsed time is trusted ahead of one that
    merely happens to hold increasing numbers. Metadata columns are excluded
    outright - a LIQ HD export offers "peak", "dc" and "duration_ms" as numeric
    columns, and any of them could otherwise be mistaken for a time axis.

    Rows are NOT required to be in order; they are sorted afterwards.

    Returns (seconds, column_name, is_clock). Seconds are absolute for a numeric
    column (so a lick file that shares a clock with the photometry stays aligned
    with it) and relative to the first timestamp for a wall-clock column, where
    the absolute epoch means nothing.
    """
    cols = list(df.columns)

    def clock_of(c):
        dt = _parse_clock(df[c])
        if dt is None:
            return None
        return dt if dt.notna().mean() > 0.8 else None

    def numeric_of(c):
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() < 0.9:
            return None
        arr = v.to_numpy(float)
        finite = arr[np.isfinite(arr)]
        if finite.size < 2 or np.all(finite == finite[0]):
            return None
        return arr

    # --- explicit override -------------------------------------------------
    if prefer_column is not None and prefer_column in cols:
        dt = clock_of(prefer_column)
        if dt is not None:
            out.start_clock = dt.dropna().iloc[0]
            sec = (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(float)
            out.notes.append(f"Using '{prefer_column}' as requested.")
            return sec, prefer_column, True
        arr = numeric_of(prefer_column)
        if arr is not None:
            f, why = _infer_unit(arr, prefer_column)
            out.notes.append(f"Using '{prefer_column}' as requested, read as {why}.")
            return arr * f, prefer_column, False
        raise ValueError(f"Column '{prefer_column}' cannot be read as a time.")

    named_clock, named_num, loose_clock, loose_num = [], [], [], []
    for c in cols:
        n = _norm(c)
        if n in _NEVER_TIME:
            continue
        dt = clock_of(c)
        if dt is not None:
            (named_clock if n in _CLOCK_NAMES else loose_clock).append((c, dt))
            continue
        arr = numeric_of(c)
        if arr is None:
            continue
        if n in _ELAPSED_NAMES:
            named_num.append((c, arr))
        else:
            # An unnamed numeric column is only a plausible clock if it holds
            # many distinct values; a column of a few repeated codes is not time.
            finite = arr[np.isfinite(arr)]
            if np.unique(finite).size >= max(20, 0.5 * finite.size):
                loose_num.append((c, arr))

    clocks = named_clock + loose_clock
    nums = named_num + loose_num

    if not clocks and not nums:
        raise ValueError(
            "No usable time column. The file needs either clock timestamps "
            "(e.g. 2026-09-16T13:11:15) or a column of elapsed times such as "
            "time_sec, elapsed_s or lick_time."
        )

    # --- cross-check a clock against a counter, and say so if they disagree --
    if clocks and nums:
        cname, dt = clocks[0]
        nname, arr = nums[0]
        clock_sec = (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(float)
        f, _why = _infer_unit(arr, nname)
        num_sec = arr * f
        cspan = float(np.nanmax(clock_sec) - np.nanmin(clock_sec))
        nspan = float(np.nanmax(num_sec) - np.nanmin(num_sec))
        agree = (cspan > 0 and nspan > 0 and 0.9 <= nspan / cspan <= 1.111) \
            or (cspan < 1 and nspan < 1)
        if agree:
            out.notes.append(
                f"'{cname}' and '{nname}' agree on a {cspan/3600:.2f} h recording.")
        else:
            out.warnings.append(
                f"'{cname}' and '{nname}' disagree about how long this recording is: "
                f"the clock says {cspan/3600:.2f} h, '{nname}' says {nspan/3600:.2f} h. "
                f"Using '{cname}', because a wall clock cannot drift, wrap or be "
                f"reset by firmware the way a free-running counter can. "
                f"If '{nname}' is in fact the correct one, select it explicitly."
            )
        out.start_clock = dt.dropna().iloc[0]
        return clock_sec, cname, True

    if clocks:
        cname, dt = clocks[0]
        out.start_clock = dt.dropna().iloc[0]
        out.notes.append(f"Times taken from the clock column '{cname}'.")
        return ((dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(float),
                cname, True)

    nname, arr = nums[0]
    f, why = _infer_unit(arr, nname)
    out.notes.append(f"Times taken from '{nname}', read as {why}.")
    return arr * f, nname, False


def _classify_columns(df, time_col):
    """Split the non-time columns into state, count, and channel roles."""
    state_cols, count_cols, channel_cols = [], [], []
    for c in df.columns:
        if c == time_col:
            continue
        n = _norm(c)
        if n in _CHANNEL_NAMES:
            channel_cols.append(c)
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().mean() < 0.5:
            continue
        vals = v.dropna().to_numpy()
        if vals.size == 0:
            continue
        uniq = np.unique(vals)
        is_binary = uniq.size <= 2 and np.all(np.isin(uniq, [0, 1]))
        looks_county = (np.all(vals >= 0) and np.all(vals == np.floor(vals))
                        and uniq.size > 2 and vals.max() <= 10000)
        if is_binary and (n in _STATE_NAMES or "lick" in n or uniq.size == 2):
            state_cols.append(c)
        elif looks_county and (n in _COUNT_NAMES or "lick" in n or "count" in n):
            count_cols.append(c)
    return state_cols, count_cols, channel_cols


def load_licks_csv(source, time_column=None, channel_column=None,
                   min_inter_lick_sec=0.05, value_column=None):
    """
    Read lick times from almost any CSV, working out the layout itself.

    source          a path, file-like object, or an existing DataFrame
    time_column     force a specific time column instead of auto-detecting
    channel_column  force the column that separates bottles
    value_column    force the state or count column
    min_inter_lick_sec  debounce; contacts closer than this count as one lick

    Returns a LickLoad. Read its .warnings before trusting the numbers.
    """
    out = LickLoad()

    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)
    if df.empty:
        raise ValueError("That CSV has no rows.")

    df.columns = [str(c).strip() for c in df.columns]
    # Drop an unnamed index column, which pandas writes by default and which
    # otherwise looks exactly like a monotonic time axis.
    drop = [c for c in df.columns if _norm(c) in ("", "unnamed0", "index")]
    if drop:
        df = df.drop(columns=drop)
        out.notes.append(f"Ignored index column(s): {', '.join(drop)}.")

    # ---- LIQ HD raw export ------------------------------------------------
    # Rows of several record types; only 'lick' rows are events.
    rec_col = next((c for c in df.columns if _norm(c) == "recordtype"), None)
    if rec_col is not None:
        kinds = df[rec_col].astype(str).str.strip().str.lower()
        session_rows = df[kinds.isin(["session_start", "sessionstart"])]
        df = df[kinds == "lick"].copy()
        out.fmt = "LIQ HD raw export"
        if df.empty:
            raise ValueError(
                "This looks like a LIQ HD export, but it contains no rows with "
                f"record_type = 'lick'. Record types present: "
                f"{', '.join(sorted(set(kinds)))}."
            )
        if not session_rows.empty:
            note = str(session_rows.iloc[0].get("note", "") or "")
            if note:
                out.notes.append(f"Session header: {note}")

    # ---- time base ---------------------------------------------------------
    t_sec, tcol, is_clock = _pick_time_base(df, out, prefer_column=time_column)
    out.time_column = tcol
    good = np.isfinite(t_sec)
    df = df.loc[good].copy()
    t_sec = t_sec[good]
    if t_sec.size == 0:
        raise ValueError(f"Column '{tcol}' produced no usable times.")

    order = np.argsort(t_sec, kind="stable")
    df = df.iloc[order]
    t_sec = t_sec[order]

    # Absolute times are preserved for a numeric column. Rebasing to the first
    # lick would shift every lick by however long the animal took to start
    # drinking, which silently misaligns the raster against the photometry.
    # A wall-clock column is the exception: its epoch carries no meaning, so it
    # is expressed as seconds from the first timestamp in the file.
    out.duration_sec = float(t_sec[-1] - min(0.0, float(t_sec[0])))
    if is_clock:
        out.duration_sec = float(t_sec[-1])

    state_cols, count_cols, chan_cols = _classify_columns(df, tcol)
    if channel_column is not None:
        chan_cols = [channel_column] if channel_column in df.columns else []
    if value_column is not None:
        state_cols = [value_column] if value_column in df.columns else []
        count_cols = [value_column] if value_column in df.columns else count_cols

    def _label(v):
        return f"bottle {v}"

    # ---- layout 1: one row per lick, optionally split by channel -----------
    # A row is an event when there is no state or count column telling us
    # otherwise. LIQ HD files always land here.
    if out.fmt == "LIQ HD raw export" or (not state_cols and not count_cols):
        if out.fmt == "unknown":
            out.fmt = "one timestamp per lick"
        if chan_cols:
            ch = df[chan_cols[0]]
            for val in pd.unique(ch):
                sel = (ch == val).to_numpy()
                out.sources[_label(val)] = _debounce(t_sec[sel], min_inter_lick_sec)
            out.notes.append(
                f"Split into {len(out.sources)} bottle(s) by '{chan_cols[0]}'.")
        else:
            out.sources["licks"] = _debounce(t_sec, min_inter_lick_sec)
        return _finalise(out, min_inter_lick_sec)

    # ---- layout 2: per-bin counts ------------------------------------------
    if count_cols:
        out.fmt = "per-bin lick counts"
        dt = float(np.median(np.diff(t_sec))) if t_sec.size > 1 else 1.0
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0
        for c in count_cols:
            counts = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
            counts = np.maximum(counts.astype(int), 0)
            parts = [t0 + (np.arange(k) + 0.5) * (dt / k)
                     for t0, k in zip(t_sec, counts) if k > 0]
            out.sources[str(c)] = (np.concatenate(parts) if parts
                                   else np.array([], float))
        out.duration_sec = float(t_sec[-1] + dt)
        out.warnings.append(
            f"This file holds lick COUNTS per {dt:g} s bin, not individual lick "
            f"times. Licks have been spread evenly inside each bin, so totals, "
            f"rates and hourly patterns are exact but inter-lick intervals and "
            f"bout structure are not. Use a file of individual lick times for "
            f"microstructure."
        )
        return _finalise(out, min_inter_lick_sec)

    # ---- layout 3: sampled 0/1 state, one column per bottle ---------------
    out.fmt = "sampled 0/1 state column" + ("s" if len(state_cols) > 1 else "")
    for c in state_cols:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy()
        s = (s > 0).astype(np.int8)
        rising = np.flatnonzero(np.diff(s) == 1) + 1
        # A trace that starts already high has its first lick at the first sample.
        if s.size and s[0] == 1:
            rising = np.concatenate([[0], rising])
        out.sources[str(c)] = _debounce(t_sec[rising], min_inter_lick_sec)
    if len(state_cols) > 1:
        out.notes.append(
            f"Found {len(state_cols)} state columns and treated each as one "
            f"bottle: {', '.join(map(str, state_cols))}.")
    return _finalise(out, min_inter_lick_sec)


def _finalise(out: LickLoad, min_gap: float) -> LickLoad:
    """Drop empty sources, and sanity-check the lick rate that came out."""
    out.sources = {k: v for k, v in out.sources.items() if len(v)}
    if not out.sources:
        out.warnings.append("No licks were found in this file.")
        return out

    all_t = out.combined
    if out.duration_sec and np.isfinite(out.duration_sec) and out.duration_sec > 0:
        per_hour = len(all_t) / (out.duration_sec / 3600.0)
        if per_hour < 1:
            out.warnings.append(
                f"Only {per_hour:.2f} licks per hour over "
                f"{out.duration_sec/3600:.1f} h. That is far below what a "
                f"drinking animal produces, which usually means the wrong time "
                f"column was used or the recording is mostly empty."
            )

    # Within a bottle, licks should look like licking.
    for name, t in out.sources.items():
        if len(t) < 20:
            continue
        d = np.diff(t)
        d = d[d > 0]
        if d.size:
            med = float(np.median(d))
            if med > 0 and 1.0 / med > 20:
                out.warnings.append(
                    f"'{name}': median inter-lick interval is {med*1000:.0f} ms "
                    f"({1/med:.1f} Hz), faster than a mouse can lick (~4-10 Hz). "
                    f"Raise the debounce above {med*1000:.0f} ms if the sensor "
                    f"is double-counting contacts."
                )
    return out


# =============================================================================
# INGESTION  (original API, unchanged so existing callers keep working)
# =============================================================================

def _debounce(times, min_gap):
    """Collapse contacts closer together than min_gap into a single lick."""
    times = np.sort(np.asarray(times, dtype=float))
    times = times[np.isfinite(times)]
    if times.size == 0 or min_gap <= 0:
        return times
    keep = [times[0]]
    for t in times[1:]:
        if t - keep[-1] >= min_gap:
            keep.append(t)
    return np.asarray(keep, dtype=float)


def licks_from_digital(digital, fs, min_inter_lick_sec=0.05, time_offset=0.0):
    """
    Lick times from a pyPhotometry digital channel.

    Takes rising edges, so one lick = one low->high transition regardless of how
    long the contact is held.
    """
    d = np.asarray(digital).astype(np.int8)
    if d.size < 2:
        return np.array([], dtype=float)
    rising = np.flatnonzero(np.diff(d) == 1) + 1
    times = rising.astype(float) / float(fs) + float(time_offset)
    return _debounce(times, min_inter_lick_sec)


def licks_from_timestamp_csv(df, time_column=None, unit="s",
                             min_inter_lick_sec=0.05):
    """Lick times from a CSV holding one timestamp per lick."""
    if time_column is None:
        for c in ["lick_time", "lick_times", "timestamp", "time_sec", "time",
                  "Time", "onset", "licks"]:
            if c in df.columns:
                time_column = c
                break
        else:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric:
                raise ValueError("No numeric column found for lick timestamps.")
            time_column = numeric[0]

    col = df[time_column]
    times = pd.to_numeric(col, errors="coerce")
    # A clock column ("13:11:15.829") is not numeric; convert it rather than
    # dropping every row and reporting zero licks.
    if times.notna().mean() < 0.5:
        dt = _parse_clock(col)
        if dt is None or dt.notna().sum() < 2:
            raise ValueError(
                f"Column '{time_column}' is neither numeric nor parseable as "
                f"clock times.")
        times = (dt - dt.dropna().iloc[0]).dt.total_seconds()
        unit = "s"
    times = times.dropna().to_numpy(float)
    if unit == "ms":
        times = times / 1000.0
    elif unit == "min":
        times = times * 60.0
    return _debounce(times, min_inter_lick_sec), time_column


def licks_from_state_csv(df, time_column, state_column, unit="s",
                         min_inter_lick_sec=0.05):
    """Lick times from a sampled binary state column (0/1 or False/True)."""
    t = pd.to_numeric(df[time_column], errors="coerce").to_numpy(float)
    if not np.isfinite(t).any():
        dt = _parse_clock(df[time_column])
        if dt is not None:
            t = (dt - dt.dropna().iloc[0]).dt.total_seconds().to_numpy(float)
            unit = "s"
    s = pd.to_numeric(df[state_column], errors="coerce").fillna(0).to_numpy()
    if unit == "ms":
        t = t / 1000.0
    elif unit == "min":
        t = t * 60.0

    good = np.isfinite(t) & np.isfinite(s)
    t, s = t[good], (s[good] > 0).astype(np.int8)
    if t.size < 2:
        return np.array([], dtype=float)
    rising = np.flatnonzero(np.diff(s) == 1) + 1
    return _debounce(t[rising], min_inter_lick_sec)


# =============================================================================
# BOUT DETECTION
# =============================================================================

def detect_bouts(lick_times, inter_bout_sec=1.0, min_licks_per_bout=3):
    """
    Group licks into bouts.

    Returns a DataFrame with one row per bout: onset, offset, duration,
    n_licks, mean within-bout lick rate (Hz), and the gap since the previous bout.
    """
    lick_times = np.sort(np.asarray(lick_times, dtype=float))
    lick_times = lick_times[np.isfinite(lick_times)]
    cols = ["bout", "onset_sec", "offset_sec", "duration_sec", "n_licks",
            "lick_rate_hz", "gap_before_sec"]
    if lick_times.size == 0:
        return pd.DataFrame(columns=cols)

    gaps = np.diff(lick_times)
    breaks = np.flatnonzero(gaps > float(inter_bout_sec))
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [lick_times.size - 1]])

    rows = []
    prev_offset = None
    for s, e in zip(starts, ends):
        n = int(e - s + 1)
        if n < int(min_licks_per_bout):
            continue
        onset, offset = float(lick_times[s]), float(lick_times[e])
        dur = offset - onset
        rows.append({
            "onset_sec": onset,
            "offset_sec": offset,
            "duration_sec": dur,
            "n_licks": n,
            # (n-1) intervals span the bout, so rate = (n-1)/duration.
            "lick_rate_hz": (n - 1) / dur if dur > 0 else np.nan,
            "gap_before_sec": np.nan if prev_offset is None else onset - prev_offset,
        })
        prev_offset = offset

    out = pd.DataFrame(rows, columns=[c for c in cols if c != "bout"])
    out.insert(0, "bout", np.arange(1, len(out) + 1))
    return out


def lick_rate_trace(lick_times, time_grid, window_sec=60.0):
    """Lick rate (Hz) on a supplied time grid, via a centred counting window."""
    time_grid = np.asarray(time_grid, dtype=float)
    lick_times = np.sort(np.asarray(lick_times, dtype=float))
    if lick_times.size == 0 or time_grid.size == 0:
        return np.zeros_like(time_grid)
    half = float(window_sec) / 2.0
    left = np.searchsorted(lick_times, time_grid - half, side="left")
    right = np.searchsorted(lick_times, time_grid + half, side="right")
    return (right - left) / float(window_sec)


# =============================================================================
# PERI-EVENT ALIGNMENT
# =============================================================================

def peri_event_matrix(time_sec, values, event_times, pre_sec=10.0, post_sec=20.0,
                      baseline_start_sec=-10.0, baseline_end_sec=-2.0,
                      normalise="baseline_z", target_dt=None):
    """
    Build a trials x time matrix of the signal aligned to each event.

    normalise
      "baseline_z" : z-score each trial against its own pre-event baseline.
                     This is the standard for event-locked photometry - the units
                     become "SDs away from this animal's own pre-event state",
                     which is comparable across trials, animals and sessions.
      "baseline_sub": subtract the baseline mean only (keeps dF/F units).
      "none"        : raw values.

    Trials whose full window falls outside the recording are dropped, so a
    partial window can never be silently padded with zeros.

    Returns (matrix, trial_time_axis, kept_event_times).
    """
    time_sec = np.asarray(time_sec, dtype=float)
    values = np.asarray(values, dtype=float)
    event_times = np.asarray(event_times, dtype=float)

    if time_sec.size < 2 or event_times.size == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)

    if target_dt is None:
        target_dt = float(np.median(np.diff(time_sec)))
    if not np.isfinite(target_dt) or target_dt <= 0:
        target_dt = 0.1

    trial_t = np.arange(-abs(pre_sec), abs(post_sec) + target_dt, target_dt)
    t_min, t_max = float(time_sec[0]), float(time_sec[-1])

    order = np.argsort(time_sec)
    ts, vs = time_sec[order], values[order]

    rows, kept = [], []
    for ev in event_times:
        if ev - abs(pre_sec) < t_min or ev + abs(post_sec) > t_max:
            continue
        trace = np.interp(ev + trial_t, ts, vs)

        if normalise in ("baseline_z", "baseline_sub"):
            bmask = (trial_t >= baseline_start_sec) & (trial_t <= baseline_end_sec)
            if bmask.sum() < 2:
                bmask = trial_t < 0
            if bmask.sum() < 2:
                continue
            if normalise == "baseline_z":
                trace = zscore(trace, baseline_mask=bmask)
            else:
                trace = trace - np.nanmean(trace[bmask])

        rows.append(trace)
        kept.append(ev)

    if not rows:
        return np.zeros((0, trial_t.size)), trial_t, np.zeros(0)
    return np.vstack(rows), trial_t, np.asarray(kept, dtype=float)


def bout_summary_table(bouts, lick_times, total_duration_sec):
    """Session-level lickometer summary."""
    n_licks = int(np.asarray(lick_times).size)
    hours = total_duration_sec / 3600.0 if total_duration_sec else np.nan
    if bouts is None or bouts.empty:
        return pd.DataFrame([{
            "Total licks": n_licks,
            "Licks per hour": round(n_licks / hours, 2) if hours else np.nan,
            "Bouts": 0, "Mean licks per bout": np.nan,
            "Mean bout duration (s)": np.nan,
            "Mean within-bout rate (Hz)": np.nan,
            "Time spent licking (%)": 0.0,
        }])
    return pd.DataFrame([{
        "Total licks": n_licks,
        "Licks per hour": round(n_licks / hours, 2) if hours else np.nan,
        "Bouts": int(len(bouts)),
        "Mean licks per bout": round(float(bouts.n_licks.mean()), 2),
        "Mean bout duration (s)": round(float(bouts.duration_sec.mean()), 2),
        "Mean within-bout rate (Hz)": round(float(bouts.lick_rate_hz.mean()), 2),
        "Time spent licking (%)": round(
            100.0 * float(bouts.duration_sec.sum()) / total_duration_sec, 2
        ) if total_duration_sec else np.nan,
    }])


# =============================================================================
# PLOTTING
#
# Plotly figures, built here rather than in app.py so the same picture can be
# produced from a script or a notebook. theme.py is imported lazily: it pulls in
# streamlit, and these functions must work without it.
# =============================================================================

_FALLBACK_COLORS = {
    "lick": "#7C3AED", "lick_rate": "#0891b2", "bout": "#f59e0b",
    "dff": "#D55E00", "z_smooth": "#0072B2",
}
_BOTTLE_COLORS = ["#7C3AED", "#0891b2", "#D55E00", "#059669",
                  "#DB2777", "#2563eb", "#B45309", "#4D7C0F",
                  "#9333EA", "#0E7490", "#B91C1C", "#65A30D"]


def _colors():
    try:
        from theme import TRACE_COLORS, style_figure
        return TRACE_COLORS, style_figure
    except Exception:
        return _FALLBACK_COLORS, (lambda fig, **kw: fig)


def _as_sources(licks):
    """Accept a LickLoad, a dict of arrays, or a bare array."""
    if isinstance(licks, LickLoad):
        return dict(licks.sources)
    if isinstance(licks, dict):
        return {k: np.asarray(v, float) for k, v in licks.items()}
    return {"licks": np.asarray(licks, float)}


def plot_licks(licks, duration_sec=None, bouts=None, rate_window_sec=60.0,
               title="Licks", height=460, max_raster_points=40000):
    """
    Lick raster over a lick-rate trace, one row per bottle.

    Down-samples the raster when a recording holds more ticks than a browser can
    draw - a 24 h file can carry 100k licks, and rendering every one makes the
    page unusable while adding nothing you can see. The rate trace below always
    uses every lick, so nothing is lost from the quantitative panel.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors, style_figure = _colors()
    sources = _as_sources(licks)
    if isinstance(licks, LickLoad) and duration_sec is None:
        duration_sec = licks.duration_sec

    all_t = np.concatenate([v for v in sources.values()]) if sources else np.array([])
    if duration_sec is None or not np.isfinite(duration_sec) or duration_sec <= 0:
        duration_sec = float(all_t.max()) if all_t.size else 1.0

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
        row_heights=[0.42, 0.58],
        subplot_titles=("Individual licks", f"Lick rate ({rate_window_sec:g} s window)"),
    )

    hours = duration_sec / 3600.0
    x_scale = 3600.0 if hours > 2 else 1.0
    x_label = "Time (h)" if x_scale == 3600.0 else "Time (s)"

    for i, (name, t) in enumerate(sources.items()):
        colour = _BOTTLE_COLORS[i % len(_BOTTLE_COLORS)]
        t = np.asarray(t, float)
        shown = t
        if t.size > max_raster_points:
            step = int(np.ceil(t.size / max_raster_points))
            shown = t[::step]
        y = np.full(shown.size, i, dtype=float)
        fig.add_trace(go.Scattergl(
            x=shown / x_scale, y=y, mode="markers",
            marker=dict(symbol="line-ns-open", size=7, color=colour,
                        line=dict(width=1.1, color=colour)),
            name=str(name), legendgroup=str(name),
            hovertemplate=f"{name}<br>%{{x:.4f}} {x_label[6:-1]}<extra></extra>",
        ), row=1, col=1)

        grid = np.linspace(0, duration_sec, min(4000, max(200, int(duration_sec / 10) + 2)))
        rate = lick_rate_trace(t, grid, rate_window_sec)
        fig.add_trace(go.Scatter(
            x=grid / x_scale, y=rate, mode="lines",
            line=dict(color=colour, width=1.6),
            name=str(name), legendgroup=str(name), showlegend=False,
            hovertemplate=f"{name}<br>%{{y:.2f}} Hz<extra></extra>",
        ), row=2, col=1)

    if bouts is not None and len(bouts):
        # Every bout as ONE trace, built from NaN-separated rectangles.
        # add_vrect per bout looks natural and is unusable in practice: a 24 h
        # recording has ~1000 bouts, and 1000 layout shapes take minutes to
        # assemble and leave the browser crawling. One filled trace draws the
        # same picture instantly.
        on = bouts["onset_sec"].to_numpy(float) / x_scale
        off = bouts["offset_sec"].to_numpy(float) / x_scale
        lo, hi = -0.6, len(sources) - 0.4
        bx = np.empty(on.size * 5)
        by = np.empty(on.size * 5)
        bx[0::5], bx[1::5], bx[2::5], bx[3::5], bx[4::5] = on, on, off, off, np.nan
        by[0::5], by[1::5], by[2::5], by[3::5], by[4::5] = lo, hi, hi, lo, np.nan
        fig.add_trace(go.Scatter(
            x=bx, y=by, fill="toself",
            fillcolor="rgba(245,158,11,0.16)", line=dict(width=0),
            mode="lines", hoverinfo="skip", showlegend=True,
            name=f"bouts ({len(bouts)})",
        ), row=1, col=1)

    labels = list(sources)
    fig.update_yaxes(
        row=1, col=1, tickmode="array", tickvals=list(range(len(labels))),
        ticktext=labels, range=[-0.6, len(labels) - 0.4], title_text="",
    )
    fig.update_yaxes(title_text="Licks / s", row=2, col=1, rangemode="tozero")
    fig.update_xaxes(title_text=x_label, row=2, col=1)
    fig.update_layout(title=title, height=height, hovermode="closest")
    style_figure(fig)
    # style_figure resets the axis titles, so re-apply them.
    fig.update_yaxes(title_text="Licks / s", row=2, col=1)
    fig.update_xaxes(title_text=x_label, row=2, col=1)
    fig.update_yaxes(
        row=1, col=1, tickmode="array", tickvals=list(range(len(labels))),
        ticktext=labels, range=[-0.6, len(labels) - 0.4],
    )
    return fig


def plot_licks_with_signal(time_sec, values, licks, value_label="dF/F (%)",
                           rate_window_sec=60.0, title="Photometry with licks",
                           height=560, max_raster_points=20000):
    """Photometry trace on top, lick raster and rate underneath, sharing an x axis."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors, style_figure = _colors()
    sources = _as_sources(licks)
    time_sec = np.asarray(time_sec, float)
    values = np.asarray(values, float)

    duration = float(time_sec[-1]) if time_sec.size else 1.0
    x_scale = 3600.0 if duration / 3600.0 > 2 else 1.0
    x_label = "Time (h)" if x_scale == 3600.0 else "Time (s)"

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.46, 0.2, 0.34],
                        subplot_titles=(value_label, "Licks", "Lick rate"))

    fig.add_trace(go.Scattergl(
        x=time_sec / x_scale, y=values, mode="lines",
        line=dict(color=colors.get("dff", "#D55E00"), width=1.2),
        name=value_label,
    ), row=1, col=1)

    for i, (name, t) in enumerate(sources.items()):
        colour = _BOTTLE_COLORS[i % len(_BOTTLE_COLORS)]
        t = np.asarray(t, float)
        shown = t[::int(np.ceil(t.size / max_raster_points))] if t.size > max_raster_points else t
        fig.add_trace(go.Scattergl(
            x=shown / x_scale, y=np.full(shown.size, i, float), mode="markers",
            marker=dict(symbol="line-ns-open", size=7, color=colour,
                        line=dict(width=1.1, color=colour)),
            name=str(name), legendgroup=str(name),
        ), row=2, col=1)
        grid = np.linspace(0, duration, min(4000, max(200, int(duration / 10) + 2)))
        fig.add_trace(go.Scatter(
            x=grid / x_scale, y=lick_rate_trace(t, grid, rate_window_sec),
            mode="lines", line=dict(color=colour, width=1.5),
            name=str(name), legendgroup=str(name), showlegend=False,
        ), row=3, col=1)

    labels = list(sources)
    fig.update_layout(title=title, height=height, hovermode="x unified")
    style_figure(fig)
    fig.update_yaxes(title_text=value_label, row=1, col=1)
    fig.update_yaxes(row=2, col=1, tickmode="array",
                     tickvals=list(range(len(labels))), ticktext=labels,
                     range=[-0.6, len(labels) - 0.4])
    fig.update_yaxes(title_text="Licks / s", row=3, col=1, rangemode="tozero")
    fig.update_xaxes(title_text=x_label, row=3, col=1)
    return fig


def plot_peri_event(matrix, trial_t, value_label="z (baseline)",
                    title="Aligned to bout onset", height=520):
    """Mean +/- SEM over trials, with the trial-by-trial heatmap beneath."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    colors, style_figure = _colors()
    matrix = np.asarray(matrix, float)
    trial_t = np.asarray(trial_t, float)
    if matrix.size == 0:
        fig = go.Figure()
        fig.add_annotation(text="No complete trials in the recording window",
                           showarrow=False)
        style_figure(fig)
        return fig

    mean = np.nanmean(matrix, axis=0)
    n = np.sum(np.isfinite(matrix), axis=0)
    sem = np.nanstd(matrix, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        row_heights=[0.5, 0.5],
                        subplot_titles=(f"Mean +/- SEM (n = {matrix.shape[0]} trials)",
                                        "Individual trials"))
    line = colors.get("z_smooth", "#0072B2")
    fig.add_trace(go.Scatter(x=np.concatenate([trial_t, trial_t[::-1]]),
                             y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                             fill="toself", fillcolor="rgba(0,114,178,0.18)",
                             line=dict(width=0), hoverinfo="skip",
                             showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=trial_t, y=mean, mode="lines",
                             line=dict(color=line, width=2), name="mean"),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(z=matrix, x=trial_t, colorscale="RdBu_r",
                             zmid=0, colorbar=dict(title=value_label, len=0.45,
                                                   y=0.22)),
                  row=2, col=1)
    fig.add_vline(x=0, line=dict(color="#111827", width=1.4, dash="dash"))
    fig.update_layout(title=title, height=height)
    style_figure(fig)
    fig.update_yaxes(title_text=value_label, row=1, col=1)
    fig.update_yaxes(title_text="Trial", row=2, col=1)
    fig.update_xaxes(title_text="Time from onset (s)", row=2, col=1)
    return fig
