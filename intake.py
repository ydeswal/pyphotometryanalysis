#!/usr/bin/env python3
"""
intake.py
=========

Everything the user drops into the main uploader - single files, or one ZIP
holding .ppd recordings, a zipped FED3_data folder, Plexon exports ... - is
unpacked and sorted here, so the app never has to guess what a file is from
its extension alone.

    expand(paths, dest)  unzip (also ZIPs inside ZIPs) into dest, safely
    classify(files)      -> dict of lists:
        photometry   .ppd files and pyPhotometry CSV recordings (Analog1, Analog2 ...)
        csv          other CSVs for the photometry pipeline (Plexon exports and their
                     event lists, CSVs already processed by this app, raw 2-channel CSVs)
        fed          FED3 controller files: FED###_<animal>_data.csv, _pc_events.csv,
                     session_info.json, alerts_log.csv
        skipped      [(name, reason)] for everything left out, with the reason

Why: before this, a FED3 *_data.csv put into the main uploader went to the
photometry CSV reader, which took Device_Number for the time column and then
failed with "Could not identify raw 405/isosbestic and 465/signal columns".

Snapshots (copies the acquisition program makes while a recording runs,
'<name>_snap-<time>.ppd' / '_autosnap-', header snapshot: true) are used only
when the finished recording is not there, and then only the newest one.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import fed3
import photometry_core as pc

MAX_MEMBER_BYTES = 4_000_000_000        # one file inside a ZIP
MAX_DEPTH = 3                           # ZIPs inside ZIPs inside ZIPs
_SNAP_RE = re.compile(r"^(?P<stem>.+?)_(?:auto)?snap-(?P<when>\d{4}-\d{2}-\d{2}-\d{6})$", re.I)
_SAFE = re.compile(r"[^A-Za-z0-9_.=+-]")


def _safe_parts(name):
    parts = [p for p in re.split(r"[\\/]", name) if p and p not in (".", "..")]
    if not parts or parts[0] == "__MACOSX" or parts[-1].startswith("._") or parts[-1] in (".DS_Store",
                                                                                           "Thumbs.db"):
        return None
    return [_SAFE.sub("_", p)[:150] for p in parts]


def expand(paths, dest, _depth=0):
    """
    Unpack ZIPs (and ZIPs inside them) into `dest`, keeping their folder names, so
    a FED3 session folder stays a folder. Other files are returned as they are.
    Member names are sanitised: nothing can be written outside `dest`.
    Returns the list of files.
    """
    dest = Path(dest)
    out = []
    for p in map(Path, paths):
        if p.is_dir():
            out += expand([f for f in sorted(p.rglob("*")) if f.is_file()], dest, _depth)
            continue
        if p.suffix.lower() != ".zip":
            out.append(p)
            continue
        if _depth >= MAX_DEPTH:
            continue
        root = dest / _SAFE.sub("_", p.stem)[:120]
        k = 1
        while root.exists() and any(root.iterdir()):
            k += 1
            root = dest / f"{_SAFE.sub('_', p.stem)[:120]}_{k}"
        nested = []
        try:
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    parts = _safe_parts(info.filename)
                    if parts is None or info.file_size > MAX_MEMBER_BYTES:
                        continue
                    target = root.joinpath(*parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(target, "wb") as dst:
                        while True:
                            chunk = src.read(8 << 20)
                            if not chunk:
                                break
                            dst.write(chunk)
                    if target.suffix.lower() == ".zip":
                        nested.append(target)
                    else:
                        out.append(target)
        except zipfile.BadZipFile:
            out.append(p)            # reported as unreadable by classify()
            continue
        for nz in nested:
            out += expand([nz], nz.parent, _depth + 1)
    return out


def _first_line(path, n=4096):
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.readline(n)
    except OSError:
        return ""


def _cols(line):
    return [c.strip().strip('"').strip() for c in line.strip().split(",")]


def csv_kind(path):
    """'fed', 'fed_alerts', 'pyphotometry', 'plexon', 'plexon_events' or 'other' for one CSV."""
    p = Path(path)
    name = p.name
    if fed3._DATA_RE.match(name) or fed3._EVENTS_RE.match(name):
        return "fed"
    if name.lower() == "alerts_log.csv":
        return "fed_alerts"
    cols = _cols(_first_line(p))
    low = {c.lower() for c in cols}
    if pc.is_pyphotometry_csv(cols):
        return "pyphotometry"
    # FED3 rows renamed by hand: the FED3 columns or the controller's pc_events columns.
    if ({"pellet_count", "event"} <= low) or ({"pc_time_unix", "event"} <= low) or \
            ({"pc_timestamp", "event", "device_number"} <= low):
        return "fed"
    import plexon
    if plexon.looks_like_photometry(cols):
        return "plexon"
    if plexon.looks_like_events(cols):
        return "plexon_events"
    return "other"


def _ppd_header(path):
    try:
        with open(path, "rb") as fh:
            return pc.parse_header_bytes(fh.read(int.from_bytes(fh.read(2), "little")))
    except Exception:
        return None


def _snapshot_key(path, header):
    """(recording name, snapshot time) for a snapshot, else None."""
    p = Path(path)
    m = _SNAP_RE.match(p.stem)
    if header and header.get("snapshot"):
        of = Path(str(header.get("snapshot_of") or (m.group("stem") if m else p.stem))).stem
        return of, str(header.get("snapshot_time") or (m.group("when") if m else ""))
    if m:
        return m.group("stem"), m.group("when")
    return None


def classify(files):
    """Sort files for the app; see the module docstring."""
    res = {"photometry": [], "csv": [], "fed": [], "skipped": []}
    files = [Path(f) for f in files]
    jsons = {f.with_suffix("").as_posix().lower(): f for f in files if f.suffix.lower() == ".json"}
    used_json = set()
    snaps = {}
    finished = set()
    for f in files:
        suf = f.suffix.lower()
        if suf == ".ppd":
            hdr = _ppd_header(f)
            if hdr is None:
                res["skipped"].append((f.name, "not a readable pyPhotometry .ppd file"))
                continue
            key = _snapshot_key(f, hdr)
            if key:
                snaps.setdefault(key[0].lower(), []).append((key[1], f))
            else:
                finished.add(f.stem.lower())
                res["photometry"].append(f)
        elif suf == ".csv":
            kind = csv_kind(f)
            if kind in ("fed", "fed_alerts"):
                res["fed"].append(f)
            elif kind == "pyphotometry":
                j = jsons.get(f.with_suffix("").as_posix().lower())
                hdr = None
                if j is not None:
                    used_json.add(j)
                    try:
                        hdr = pc.parse_header_bytes(j.read_bytes())
                    except Exception:
                        hdr = None
                key = _snapshot_key(f, hdr)
                if key:
                    snaps.setdefault(key[0].lower(), []).append((key[1], f))
                else:
                    finished.add(f.stem.lower())
                    res["photometry"].append(f)
            else:
                res["csv"].append(f)
        elif suf == ".json":
            continue
        elif suf == ".zip":
            res["skipped"].append((f.name, "ZIP could not be opened (damaged, or nested too deep)"))
        elif suf in (".xlsx", ".xls"):
            res["skipped"].append((f.name, "Excel copy; the FED3 controller's CSV files hold the same data "
                                           "and are used instead"))
        elif suf in (".png", ".jpg", ".jpeg", ".pdf", ".txt", ".md", ".log", ".gz"):
            res["skipped"].append((f.name, "not a data file the app reads"))
        else:
            res["skipped"].append((f.name, f"unknown file type '{suf or 'none'}'"))
    for f in files:
        if f.suffix.lower() == ".json" and f not in used_json:
            if f.name.lower() == "session_info.json":
                res["fed"].append(f)
            elif not f.name.lower().endswith("_clock_report.json"):
                res["skipped"].append((f.name, "JSON that is not a FED3 session_info.json or the header of a "
                                               "pyPhotometry CSV recording"))
    for rec, lst in snaps.items():
        lst.sort(key=lambda x: x[0])
        if rec in finished:
            for _, f in lst:
                res["skipped"].append((f.name, "snapshot taken during the recording; the finished recording "
                                               "is also here and is used instead"))
            continue
        res["photometry"].append(lst[-1][1])
        for _, f in lst[:-1]:
            res["skipped"].append((f.name, "older snapshot; the newest snapshot of this recording is used"))
    res["photometry"].sort(key=lambda p: fed3._natural(p.name))
    return res


def pairing(photometry, fed_animals):
    """{recording stem: FED animal or None}, using fed3.guess_animal (ID inside the name)."""
    out = {}
    for f in photometry:
        hdr = _ppd_header(f) if Path(f).suffix.lower() == ".ppd" else None
        out[Path(f).stem] = fed3.guess_animal(Path(f).stem, fed_animals, (hdr or {}).get("subject_ID", ""))
    return out


def summary_lines(res, fed_animals=None):
    """Short plain-language lines describing what was found."""
    lines = []
    if res["photometry"]:
        lines.append(f"{len(res['photometry'])} photometry recording(s)")
    plx = [f for f in res["csv"] if csv_kind(f) == "plexon"]
    if plx:
        lines.append(f"{len(plx)} Plexon recording(s)")
    other = len(res["csv"]) - len(plx)
    if other:
        lines.append(f"{other} other CSV file(s)")
    if res["fed"]:
        lines.append(f"{len(res['fed'])} FED3 file(s)"
                     + (f" ({len(fed_animals)} animal(s): {', '.join(fed_animals)})" if fed_animals else ""))
    return lines


__all__ = ["expand", "classify", "csv_kind", "pairing", "summary_lines"]
