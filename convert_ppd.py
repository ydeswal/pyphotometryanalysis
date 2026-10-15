#!/usr/bin/env python3
"""
convert_ppd.py - turn a PC-timestamped .ppd into a standard pyPhotometry .ppd
plus a timestamps file, so ANY reader can open it.

Why this exists
---------------
The modified acquisition program writes each sample as the usual 4 data words
(8 bytes) FOLLOWED BY an 8-byte PC timestamp, 16 bytes in all. Every standard
reader - the official pyPhotometry importer, GuPPy, older copies of this app -
assumes 8 bytes per sample. It reads each timestamp as two more samples of
light, so the recording comes out twice as long, the traces turn into a
sawtooth, and the 405 fit and clipping check are nonsense.

Nothing is lost in such a file; it only needs reading correctly. This script
writes, next to the original (which is never modified):

    <name>_standard.ppd          standard 8-byte-per-sample file, same data
    <name>_pc_timestamps.csv.gz  sample, pc_unix: every raw PC timestamp
    <name>_clock_report.json     start time, drift, gaps, end_time cross-check

The standard file's header keeps the PC clock information (start time of the
first sample, UTC offset), so this app still shows wall-clock times for it.

Usage
-----
    python3 convert_ppd.py m10_fasted-2026-10-14-102300.ppd
    python3 convert_ppd.py *.ppd                     # whole folder
    python3 convert_ppd.py file.ppd --outdir converted
    python3 convert_ppd.py file.ppd --no-timestamps  # skip the .csv.gz

Put this file next to photometry_core.py.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np
import pandas as pd

import photometry_core as pc


def _hms(seconds):
    seconds = int(round(float(seconds)))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def convert(path, outdir=None, write_timestamps=True, clock_source="sample"):
    path = Path(path)
    outdir = Path(outdir) if outdir else path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    with open(path, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(2), "little")).decode("utf-8"))
        buf = f.read()

    fs = float(header["sampling_rate"])
    n_analog = pc._header_int(header, "n_analog_channels", "n_analog_signals", default=2)
    version = pc._parse_version(header.get("version", "0.0.0"))
    pulsed = "pulsed" in str(header.get("mode", "")).lower()
    wpf = 2 * n_analog if (pulsed and version >= (1, 1)) else n_analog

    layout = pc._pc_layout(header, buf, wpf, fs)
    print("=" * 70)
    print(path.name)
    print("=" * 70)
    if layout is None:
        print("  This is already a standard .ppd (no per-sample PC timestamps).")
        print("  Nothing to convert.\n")
        return None

    bps, wps, how = layout
    n = len(buf) // bps
    rec = np.frombuffer(buf, dtype=np.dtype({
        "names": ["w", "t"], "formats": [("<u2", (wps,)), "<f8"],
        "offsets": [0, 2 * wps], "itemsize": bps}), count=n)
    print(f"  layout       {bps} bytes/sample = {wps} data words + 8-byte PC timestamp ({how})")
    if len(buf) % bps:
        print(f"  NOTE         dropped {len(buf) % bps} trailing bytes of an incomplete sample")

    clock = pc.build_clock_model(header, fs, n, rec["t"], clock_source=clock_source)
    print(f"  samples      {n:,} at {fs:g} Hz = {_hms(clock['duration_sec'])} "
          f"({clock['duration_sec'] / 3600:.2f} h)")
    print(f"  a standard reader would have shown {_hms(n * bps / (2 * wps) / fs)} "
          f"(x{bps / (2 * wps):g}) of garbage")
    for line in clock["notes"]:
        print("  clock       ", line)
    for line in clock["warnings"]:
        print("  WARNING     ", line)

    # ---- standard .ppd -----------------------------------------------------
    new_header = {k: v for k, v in header.items()
                  if k not in ("pc_timestamps", "bytes_per_sample", "words_per_sample")}
    new_header["converted_from_pc_timestamped_ppd"] = path.name
    if clock.get("start_unix") is not None:
        # Read back by photometry_core so wall-clock times survive conversion.
        new_header["first_sample_unix"] = clock["start_unix"]
        new_header["first_sample_local"] = clock["start_local"]
        new_header["utc_offset_s"] = clock.get("utc_offset_s")
        new_header["pc_clock_drift_ppm"] = clock.get("pc_drift_ppm")
        new_header["lost_sample_gaps"] = clock.get("gaps")
    hb = json.dumps(new_header).encode("utf-8")
    if len(hb) > 65535:
        raise ValueError("Converted header too large for the .ppd format.")
    out_ppd = outdir / f"{path.stem}_standard.ppd"
    with open(out_ppd, "wb") as f:
        f.write(len(hb).to_bytes(2, "little"))
        f.write(hb)
        f.write(np.ascontiguousarray(rec["w"]).tobytes())
    print(f"  wrote        {out_ppd}")

    # ---- timestamps sidecar -------------------------------------------------
    if write_timestamps:
        # Only the raw stamps are stored: they are the one thing the standard
        # .ppd cannot reproduce. time_sec / clock_time for every sample follow
        # from the header (first_sample_unix + sample / rate) and are written
        # by the app and ana.py into every processed CSV anyway.
        out_ts = outdir / f"{path.stem}_pc_timestamps.csv.gz"
        pd.DataFrame({"sample": np.arange(n, dtype=np.int64),
                      "pc_unix": np.asarray(rec["t"], dtype=np.float64)}).to_csv(
            out_ts, index=False, float_format="%.6f",
            compression={"method": "gzip", "compresslevel": 6})
        print(f"  wrote        {out_ts}")

    report = {k: v for k, v in clock.items()}
    report.update({"source_file": path.name, "standard_file": out_ppd.name,
                   "bytes_per_sample_original": bps, "n_samples": n,
                   "sampling_rate": fs})
    out_json = outdir / f"{path.stem}_clock_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=str))
    print(f"  wrote        {out_json}")

    # ---- verify -------------------------------------------------------------
    a = pc.read_ppd(path)
    b = pc.read_ppd(out_ppd)
    same = all(np.array_equal(a[k], b[k]) for k in a
               if k.startswith(("analog_", "digital_")) and isinstance(a[k], np.ndarray))
    print(f"  verified     converted file reads back identical: {'YES' if same else 'NO'}")
    print()
    return out_ppd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ppd", nargs="*")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--no-timestamps", action="store_true",
                    help="do not write the per-sample timestamps .csv.gz")
    ap.add_argument("--clock", choices=["sample", "pc"], default="sample",
                    help="time axis for time_sec/clock_time in the sidecar (default: sample)")
    args = ap.parse_args()
    paths = args.ppd or sorted(p for p in os.listdir(".") if p.lower().endswith(".ppd")
                               and not p.lower().endswith("_standard.ppd"))
    if not paths:
        print("Usage: python3 convert_ppd.py yourfile.ppd")
        sys.exit(1)
    for p in paths:
        if os.path.exists(p):
            convert(p, args.outdir, not args.no_timestamps, args.clock)
        else:
            print(f"File not found: {p}\n")


if __name__ == "__main__":
    main()
