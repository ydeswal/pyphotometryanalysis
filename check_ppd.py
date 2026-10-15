#!/usr/bin/env python3
"""
check_ppd.py - report how a .ppd file is laid out and whether the old reader
would have misread it.

Usage:
    python3 check_ppd.py 7431A_2days.ppd
    python3 check_ppd.py *.ppd

Put this file in the same folder as photometry_core.py, or run it from that
folder and pass a full path to the .ppd file.
"""

import sys
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import numpy as np
    import photometry_core as pc
except ImportError as e:
    missing = str(e).split("'")[-2] if "'" in str(e) else str(e)
    print(f"\nCould not import '{missing}'.\n")
    if missing == "photometry_core":
        print("photometry_core.py must sit in the same folder as this script.")
        print(f"This script is in: {HERE}")
    else:
        print("Install the dependencies first:")
        print("    python3 -m pip install numpy pandas scipy")
    sys.exit(1)


def hms(seconds):
    seconds = int(round(float(seconds)))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def check(path):
    print("=" * 68)
    print(os.path.basename(path))
    print("=" * 68)

    try:
        d = pc.read_ppd(path)
    except Exception as e:
        print(f"  Could not read this file: {e}\n")
        return

    fs = d["sampling_rate"]
    n = len(d["analog_1"])
    clock = d.get("clock") or {}
    duration = clock.get("duration_sec", n / fs)

    print(f"  subject            {d.get('subject_ID', '')}")
    print(f"  mode               {d['mode']}")
    print(f"  pyPhotometry ver.  {d['version']}")
    print(f"  sampling rate      {fs:g} Hz")
    print(f"  samples/channel    {n:,}")
    print(f"  duration           {hms(duration)}  ({duration / 3600:.2f} h)")
    print()
    print(f"  FRAME LAYOUT       {d['frame_layout']}")
    print()

    print(f"  FILE LAYOUT        {d.get('layout_note', '')}")
    print()

    # --- clock ---------------------------------------------------------------
    if clock.get("start_local"):
        print(f"  started            {str(clock['start_local'])[:23].replace('T', ' ')} (local)")
    for line in clock.get("notes", []):
        print(f"  clock              {line}")
    for line in clock.get("warnings", []):
        print(f"  WARNING            {line}")
    print()

    # --- would the old reader have got this right? ---------------------------
    if clock.get("has_pc_timestamps"):
        print("  VERDICT: every standard reader MISREADS this file, including the")
        print("    old app and the official pyPhotometry importer.")
        print("    Each sample carries an 8-byte PC timestamp after its data words,")
        print("    which a standard reader takes for two more samples of light:")
        print(f"    - it reports {hms(duration * 2)} instead of {hms(duration)}")
        print("    - the timestamps' bytes appear as a sawtooth, fake 'clipping'")
        print("      and a negative 405 slope")
        print("    This version reads it correctly. To give the file to any other")
        print("    software, run:  python3 convert_ppd.py " + os.path.basename(path))
    elif d["frame_layout"] == "pulsed_v1.1_led_on_off":
        print("  VERDICT: the old code read this file correctly.")
        print("    It assumed 4 words per frame, which is right for pulsed")
        print("    v1.1+ files. Your channel assignment was fine.")
        print("    dF/F values still change, because F0 now comes from the 465")
        print("    channel instead of the 405, but the underlying traces were")
        print("    being read properly.")
    else:
        old_n = n // 2
        print("  VERDICT: the old code MISREAD this file.")
        print("    A frame here is 2 words wide, but the old reader assumed 4,")
        print("    so it interleaved consecutive timepoints.")
        print(f"    - it would have reported {old_n:,} samples "
              f"({hms(old_n / fs)}) instead of {hms(duration)}")
        print("    - 465 and 405 would have collapsed into the same quantity")
        print("    Anything previously analysed from this file needs redoing.")
    print()

    # --- channel sanity ------------------------------------------------------
    a1, a2 = d["analog_1"], d["analog_2"]
    r = float(np.corrcoef(a1, a2)[0, 1])
    print(f"  465 (analog_1)     mean {np.mean(a1):+.4f} V   "
          f"range [{np.min(a1):+.4f}, {np.max(a1):+.4f}]")
    print(f"  405 (analog_2)     mean {np.mean(a2):+.4f} V   "
          f"range [{np.min(a2):+.4f}, {np.max(a2):+.4f}]")
    print(f"  correlation        r = {r:.4f}")
    if r > 0.999:
        print("    WARNING: r above 0.999 means the two channels are effectively")
        print("    identical. That is the signature of a de-interleaving problem.")
    print()

    # --- digital inputs (lickometer) ----------------------------------------
    any_digital = False
    for ch in ("digital_1", "digital_2"):
        if ch in d:
            pulses = pc.digital_pulse_times(d[ch], fs, clock=clock or None)
            if len(pulses):
                any_digital = True
                rate = len(pulses) / (duration / 3600.0)
                print(f"  {ch}          {len(pulses):,} pulses  "
                      f"({rate:.1f}/hour)")
                if len(pulses) > 1:
                    ipi = np.diff(pulses)
                    med = float(np.median(ipi))
                    print(f"                     median interval {med:.3f} s"
                          f"  ({1/med:.1f} Hz)" if med > 0 else "")
    if not any_digital:
        print("  digital inputs     no pulses found on either channel")
        print("    If your lickometer was wired to a digital input, check the")
        print("    wiring. Otherwise use the CSV route in the lickometer section.")
    print()


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        paths = [f for f in sorted(os.listdir(".")) if f.lower().endswith(".ppd")]
        if not paths:
            print("Usage: python3 check_ppd.py yourfile.ppd")
            sys.exit(1)
        print(f"No file given; checking {len(paths)} .ppd file(s) in this folder.\n")

    for p in paths:
        if os.path.exists(p):
            check(p)
        else:
            print(f"File not found: {p}\n")
