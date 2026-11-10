#!/usr/bin/env python3
"""
fed3.py
=======

Read the output of the FED3 PC-control app and line it up with photometry.

Input: any mix of
  * the zipped FED3_data folder (one sub-folder per session, e.g.
    longi_20261015_103500/, refeed_20261015_193635/),
  * session folders, or loose files:
      FED###_<animal>_data.csv        one row per FED event (pellet taken, pokes)
      FED###_<animal>_pc_events.csv   pellet DISPENSED times, commands, start/stop
      session_info.json               start/stop, UTC offset, port -> animal

Animals are identified by the <animal> part of the file name (m10, m11 ...),
not by the FED device number: several FEDs can share device number 0, and
the same FED can move between animals.

Time base. Every event time is the FED's PC_Timestamp: the PC wall-clock time
at the moment of the event, computed on the FED from a sync the PC sends
every 10 s. The photometry clock_time comes from the same PC clock. Both are
handled as naive LOCAL times, which is what both files store.

Things this reader deals with because they are in the real data
---------------------------------------------------------------
* Pellet_Count restarts whenever a new SD file starts (and at every session),
  so intake is counted from Pellet EVENTS, never from the Pellet_Count column.
* A NEWFILE request waits while a pellet sits in the well, so a session's
  first row can be the last pellet of the previous file (counts from the old
  file, retrieval "Timed_out"). It is still a real pellet taken at that time.
* Two FEDs labelled with the same animal write into the same files. Rows can
  then be interleaved, and a line can be cut off where one writer overwrote
  the other. Damaged lines are salvaged where the time and event survive,
  otherwise skipped, and every case is reported.
* Periods with no FED logging (between sessions) are not "no eating"; they
  are reported as coverage gaps and shown as such.

FED health (controller 2.0 / firmware PCCtrl-2.0)
-------------------------------------------------
* The firmware writes status rows into the same data file: JamClearing,
  JamCleared, Jammed, FeedAborted, LeftBlocked/RightBlocked,
  LeftUnblocked/RightUnblocked, SDError/SDRecovered, Restart_<cause>. They are
  kind 'health' (never pokes or pellets) and are turned into intervals by
  health_intervals() and counted by health_summary().
* USB disconnects (DISCONNECTED -> CONNECTED in _pc_events.csv) are cut out of
  the coverage: nothing reached the PC files then, so it is "not logged",
  not "not eating". They are also health intervals.
* Right after a FED restart the PC clock sync is missing for a few seconds and
  PC_Timestamp is NaN; those rows use PC_Received (the PC's receive time).
* alerts_log.csv written by the email-alert program is read into .alerts.
* Files named FED003_m10_COM5_data.csv (the controller adds the port when two
  FEDs share number + ID) belong to animal m10.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PELLET_MG_DEFAULT = 20.0                 # standard FED3 dustless precision pellet

_DATA_RE = re.compile(r"^FED(\d{3})_(.+?)_data\.csv$", re.I)
_EVENTS_RE = re.compile(r"^FED(\d{3})_(.+?)_pc_events\.csv$", re.I)
_ANIMAL_RE = re.compile(r"(?<![A-Za-z0-9])([A-Za-z]{1,3}\d{1,4})(?=[^0-9]|$)")
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?$")
_PC_EVENT_NAMES = {"RECORDING_START", "RECORDING_STOP", "PELLET_DISPENSED", "COMMAND",
                   "SD_NEWFILE", "CONNECTED", "DISCONNECTED", "RECONNECTED", "TTL", "PULSE",
                   "TTL_PULSE", "FED_REBOOT", "FED_ERROR"}
_PORT_SUFFIX_RE = re.compile(r"_(COM\d+|SIM\d+|dev-[A-Za-z0-9.-]+)$", re.I)

# FED health rows written by firmware PCCtrl-2.0 (exact event names)
HEALTH_EVENTS = {"JamClearing", "JamCleared", "Jammed", "FeedAborted", "LeftBlocked", "RightBlocked",
                 "LeftUnblocked", "RightUnblocked", "SDError", "SDRecovered"}
HEALTH_LABELS = {
    "jam": "Clearing a jam",
    "jammed": "JAMMED (no pellet after 100 turns)",
    "disconnected": "USB disconnected",
    "blocked_left": "Left poke blocked",
    "blocked_right": "Right poke blocked",
    "sd_error": "SD card error",
}
RESTART_WHY = {
    "Watchdog": "froze; the watchdog restarted it",
    "Brownout": "supply voltage dropped",
    "Power": "power lost / switched off and on",
    "Software": "restarted by command or firmware upload",
    "ResetButton": "reset button pressed",
}

__all__ = ["FedData", "load_fed", "animal_from_name", "intake_bins", "find_meals",
           "rolling_rate", "window_means", "aligned_matrix", "permutation_test",
           "to_seconds", "PELLET_MG_DEFAULT", "health_intervals", "health_points",
           "health_summary", "HEALTH_LABELS", "is_health_event"]


def is_health_event(name):
    """True for the FED-status rows of firmware 2.0 (never a poke or a pellet)."""
    n = str(name).strip()
    return n in HEALTH_EVENTS or n.startswith("Restart_")


def animal_from_name(name):
    """'m10_fasted_feds-2026-10-15-103416' -> 'm10'. Case-insensitive, or None."""
    for m in _ANIMAL_RE.finditer(str(name)):
        tok = m.group(1).lower()
        if not tok.startswith("fed"):          # FED000 is the device, not the animal
            return tok
    return None


def guess_animal(name, animals, subject_id=""):
    """
    Which FED animal a recording belongs to. First any FED animal ID that
    appears in the recording name (or subject ID) as a whole token, longest
    first so 'm1' never takes 'm11'; then the usual ID guess. Case-insensitive.
    Returns the FED's spelling of the ID, or None.
    """
    canon = {str(a).lower(): a for a in animals}
    for text in (str(name), str(subject_id or "")):
        low = text.lower()
        for a in sorted(canon, key=len, reverse=True):
            if re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", low):
                return canon[a]
    for text in (str(name), str(subject_id or "")):
        g = animal_from_name(text)
        if g and g in canon:
            return canon[g]
    return None


def _ts(text):
    if text is None:
        return pd.NaT
    s = str(text).strip()
    if not s or s.lower() == "nan":
        return pd.NaT
    try:
        return pd.Timestamp(s)
    except (ValueError, TypeError):
        return pd.NaT


@dataclass
class FedData:
    """
    events    one row per event: animal, session, kind ('pellet' | 'poke' |
              'dispensed' | 'other'), event (raw name), time (local
              Timestamp), fed (device number), retrieval_s, ipi_s,
              poke_time_s, file
    coverage  when each animal's FED was being logged: animal, session,
              start, stop
    sessions  session, start, stop, utc_offset_h, animals
    health    FED status changes: animal, session, event, time, detail. Built
              from the 'health' data rows and the controller's connect /
              disconnect / reboot records; see health_intervals()
    alerts    alerts_log.csv rows: animal, session, time, rule, message
    """
    events: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)
    sessions: pd.DataFrame = field(default_factory=pd.DataFrame)
    health: pd.DataFrame = field(default_factory=pd.DataFrame)
    alerts: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def animals(self):
        if self.events.empty:
            return []
        return sorted(self.events["animal"].dropna().unique(), key=_natural)

    def times(self, animal, kind="pellet", sessions=None):
        e = self.events
        m = (e["animal"] == animal) & (e["kind"] == kind)
        if sessions is not None:
            m &= e["session"].isin(list(sessions))
        return e.loc[m, "time"].sort_values().reset_index(drop=True)

    def coverage_of(self, animal, sessions=None):
        c = self.coverage
        if c.empty:
            return c
        m = c["animal"] == animal
        if sessions is not None:
            m &= c["session"].isin(list(sessions))
        return c.loc[m].sort_values("start").reset_index(drop=True)


def _natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


# =============================================================================
# READING
# =============================================================================

def _collect(paths, workdir):
    """
    Expand zips / folders / files into {session_folder_name: [Path, ...]}.
    Zip members are extracted under `workdir` keeping only their last folder
    and file name, so a crafted archive cannot write outside it.
    """
    workdir = Path(workdir)
    groups = {}

    def add(p, session):
        groups.setdefault(session, []).append(Path(p))

    for p in map(Path, paths):
        if p.is_dir():
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in (".csv", ".json"):
                    add(f, f.parent.name or "uploaded")
        elif p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as z:
                for info in z.infolist():
                    if info.is_dir():
                        continue
                    parts = [s for s in re.split(r"[\\/]", info.filename) if s and s not in (".", "..")]
                    if not parts or parts[0] == "__MACOSX" or parts[-1].startswith("._"):
                        continue
                    if not parts[-1].lower().endswith((".csv", ".json")):
                        continue
                    session = parts[-2] if len(parts) > 1 else p.stem
                    safe_s = re.sub(r"[^A-Za-z0-9_.-]", "_", session)[:120]
                    safe_f = re.sub(r"[^A-Za-z0-9_.=-]", "_", parts[-1])[:160]
                    out = workdir / safe_s / safe_f
                    out.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, open(out, "wb") as dst:
                        dst.write(src.read(50_000_000))      # 50 MB cap per member
                    add(out, safe_s)
        elif p.is_file():
            add(p, p.parent.name or "uploaded")
    return groups


def _session_meta(name, files, out):
    info = {"session": name, "start": pd.NaT, "stop": pd.NaT, "utc_offset_h": None,
            "ports": {}}
    m = re.match(r"^(.*?)_(\d{8})_(\d{6})$", name)
    info["label"] = m.group(1) if m else name
    if m:
        info["start"] = pd.to_datetime(m.group(2) + m.group(3), format="%Y%m%d%H%M%S", errors="coerce")
    sj = next((f for f in files if f.name.lower() == "session_info.json"), None)
    if sj is not None:
        try:
            js = json.loads(sj.read_text(encoding="utf-8", errors="replace"))
            info["start"] = _ts(js.get("start_pc_time")) if js.get("start_pc_time") else info["start"]
            info["stop"] = _ts(js.get("stop_pc_time"))
            info["utc_offset_h"] = js.get("utc_offset_hours")
            info["label"] = js.get("session_name") or info["label"]
            for port, f in (js.get("feds") or {}).items():
                info["ports"][port] = {"animal": str(f.get("animal_id", "")).lower(),
                                       "fed": f.get("fed_number"),
                                       "battery": (f.get("last_status") or {}).get("battery")}
        except (ValueError, OSError) as e:
            out.warnings.append(f"{name}: session_info.json could not be read ({e}).")
    by_animal = {}
    for port, d in info["ports"].items():
        by_animal.setdefault(d["animal"], []).append(port)
    for animal, ports in by_animal.items():
        if len(ports) > 1 and animal not in ("", "na", "none"):
            out.warnings.append(
                f"Session '{name}': {len(ports)} FEDs ({', '.join(ports)}) were all labelled "
                f"'{animal}', so they wrote into the SAME files. Their rows are merged and "
                "cannot be told apart reliably (battery voltage is the only clue). Give "
                "every FED its own animal ID in the FED3 controller.")
    return info


def _read_data_csv(path, animal, fed, session, out):
    damaged = []

    def bad(line):
        damaged.append(line)
        return None
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines=bad, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if damaged:
        out.warnings.append(f"{session}/{path.name}: {len(damaged)} damaged line(s) skipped.")
    if df.empty:
        return pd.DataFrame()
    tcol = next((c for c in ("PC_Timestamp", "PC_Received") if c in df.columns), None)
    time_src = pd.Series("PC_Timestamp", index=df.index)
    if tcol is not None:
        t = pd.to_datetime(df[tcol], errors="coerce", format="mixed")
        if tcol != "PC_Timestamp":
            out.notes.append(f"{session}/{path.name}: no PC_Timestamp column; used {tcol}.")
            time_src[:] = tcol
        elif "PC_Received" in df.columns and t.isna().any():
            # e.g. the first seconds after a FED restart, before the PC re-sent its clock
            miss = t.isna()
            t2 = pd.to_datetime(df.loc[miss, "PC_Received"], errors="coerce", format="mixed")
            t[miss] = t2
            time_src[miss] = "PC_Received"
            n_fb = int(t2.notna().sum())
            if n_fb:
                out.notes.append(f"{session}/{path.name}: {n_fb} row(s) had no PC_Timestamp (FED clock "
                                 "not yet synced, e.g. just after a restart); used the PC receive time "
                                 "instead (normally within a few ms).")
    else:
        rtc = next((c for c in df.columns if c.lower().startswith("mm:dd:yyyy")), None)
        t = pd.to_datetime(df[rtc], format="%m/%d/%Y %H:%M:%S", errors="coerce") if rtc else pd.Series(pd.NaT, index=df.index)
        out.warnings.append(f"{session}/{path.name}: no PC timestamp; used the FED's own clock "
                            "(1 s resolution, only as accurate as its last sync).")
    ev = df.get("Event", pd.Series("", index=df.index)).astype(str).str.strip()
    health = ev.map(is_health_event).to_numpy(bool)
    kind = np.where(health, "health",
                    np.where(ev.str.lower() == "pellet", "pellet",
                             np.where(ev.str.contains("Left|Right|Poke", case=False), "poke", "other")))

    def num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    res = pd.DataFrame({
        "animal": animal, "session": session, "kind": kind, "event": ev, "time": t,
        "fed": int(fed), "retrieval_s": num("Retrieval_Time"), "ipi_s": num("InterPelletInterval"),
        "poke_time_s": num("Poke_Time"), "battery": num("Battery_Voltage"),
        "file": path.name, "time_source": time_src.to_numpy(),
    })
    n_bad_t = int(res["time"].isna().sum())
    if n_bad_t:
        out.warnings.append(f"{session}/{path.name}: {n_bad_t} row(s) without a readable time skipped.")
    return res.dropna(subset=["time"])


def _read_pc_events(path, animal, fed, session, out):
    """Line-by-line: the file can contain lines cut off by a second writer."""
    rows, salvaged, skipped = [], 0, 0
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in text[1:]:
        if not line.strip():
            continue
        try:
            f = next(csv.reader([line]))
        except csv.Error:
            f = line.split(",")
        ok = len(f) >= 3 and _TS_RE.match(f[0].strip() or "") and f[2].strip() in _PC_EVENT_NAMES
        if not ok:
            skipped += 1
            continue
        if len(f) != 5:
            salvaged += 1
        event = f[2].strip()
        when = _ts(f[0])
        if event == "PELLET_DISPENSED" and len(f) == 5 and _TS_RE.match(f[4].strip() or ""):
            when = _ts(f[4])                       # the FED's own PC-synced time
        rows.append({"event": event, "time": when, "detail": f[3] if len(f) > 3 else ""})
    if salvaged or skipped:
        out.warnings.append(
            f"{session}/{path.name}: {salvaged + skipped} damaged line(s) "
            f"({salvaged} recovered from their time stamp, {skipped} unreadable and skipped). "
            "This happens when two FEDs with the same animal ID write one file.")
    df = pd.DataFrame(rows)
    if df.empty:
        return df, None, pd.DataFrame()
    hl = df[df["event"].isin(["CONNECTED", "DISCONNECTED", "RECONNECTED", "FED_REBOOT", "FED_ERROR",
                              "RECORDING_START", "RECORDING_STOP"])]
    health = pd.DataFrame({"animal": animal, "session": session, "event": hl["event"].to_numpy(),
                           "time": hl["time"].to_numpy(), "detail": hl["detail"].astype(str).to_numpy(),
                           "source": "pc_events"})
    disp = df[df["event"] == "PELLET_DISPENSED"]
    ev = pd.DataFrame({"animal": animal, "session": session, "kind": "dispensed",
                       "event": "PELLET_DISPENSED", "time": disp["time"].to_numpy(),
                       "fed": int(fed), "file": path.name})
    starts = df.loc[df["event"] == "RECORDING_START", "time"]
    stops = df.loc[df["event"] == "RECORDING_STOP", "time"]
    span = (starts.min() if len(starts) else pd.NaT, stops.max() if len(stops) else pd.NaT)
    return ev, span, health


def _read_alerts(path, session, out):
    try:
        a = pd.read_csv(path, dtype=str, on_bad_lines="skip", engine="python")
    except (pd.errors.EmptyDataError, OSError, ValueError) as e:
        out.warnings.append(f"{session}/{path.name}: could not be read ({e}).")
        return pd.DataFrame()
    if a.empty or "PC_Time" not in a.columns:
        return pd.DataFrame()
    return pd.DataFrame({
        "animal": a.get("Animal", pd.Series("", index=a.index)).fillna("").str.lower().to_numpy(),
        "fed": a.get("FED", pd.Series("", index=a.index)).fillna("").to_numpy(),
        "session": session,
        "time": pd.to_datetime(a["PC_Time"], errors="coerce", format="mixed").to_numpy(),
        "rule": a.get("Rule", pd.Series("", index=a.index)).fillna("").to_numpy(),
        "message": a.get("Message", pd.Series("", index=a.index)).fillna("").to_numpy(),
    }).dropna(subset=["time"])


def _animal_of(raw, out, fname):
    """'m10' from 'm10', and from 'm10_COM5' (port added by the controller to avoid a clash)."""
    a = raw.lower()
    m = _PORT_SUFFIX_RE.search(raw)
    if m:
        a = raw[:m.start()].lower()
        out.notes.append(f"{fname}: the controller added the port ({m.group(1)}) to this file name because "
                         f"another FED had the same number and ID; it is read as animal {a}.")
    return a


def load_fed(paths, workdir):
    """Read everything given (zips, folders, files). Returns FedData."""
    out = FedData()
    groups = _collect(paths, workdir)
    ev_parts, cov_rows, sess_rows, hl_parts, al_parts = [], [], [], [], []
    for session, files in sorted(groups.items()):
        meta = _session_meta(session, files, out)
        animals_here = set()
        for f in files:
            if f.name.lower() == "alerts_log.csv":
                a = _read_alerts(f, session, out)
                if not a.empty:
                    al_parts.append(a)
                continue
            m = _DATA_RE.match(f.name)
            if m:
                fed, animal = m.group(1), _animal_of(m.group(2), out, f.name)
                if animal in ("na", "none", ""):
                    continue
                d = _read_data_csv(f, animal, fed, session, out)
                if not d.empty:
                    ev_parts.append(d)
                animals_here.add(animal)
                continue
            m = _EVENTS_RE.match(f.name)
            if m:
                fed, animal = m.group(1), _animal_of(m.group(2), out, f.name)
                if animal in ("na", "none", ""):
                    continue
                d, span, hl = _read_pc_events(f, animal, fed, session, out)
                if d is not None and not d.empty:
                    ev_parts.append(d)
                if hl is not None and not hl.empty:
                    hl_parts.append(hl)
                animals_here.add(animal)
                if span is not None:
                    cov_rows.append({"animal": animal, "session": session,
                                     "start": span[0], "stop": span[1]})
        if not animals_here:
            continue                           # no FED files in this folder
        sess_rows.append({"session": session, "label": meta["label"], "start": meta["start"],
                          "stop": meta["stop"], "utc_offset_h": meta["utc_offset_h"],
                          "animals": ", ".join(sorted(animals_here, key=_natural))})
        # Coverage falls back to the session's start/stop, then to its events.
        for a in animals_here:
            have = [r for r in cov_rows if r["animal"] == a and r["session"] == session]
            if have:
                r = have[-1]
                r["start"] = r["start"] if pd.notna(r["start"]) else meta["start"]
                r["stop"] = r["stop"] if pd.notna(r["stop"]) else meta["stop"]
            else:
                cov_rows.append({"animal": a, "session": session,
                                 "start": meta["start"], "stop": meta["stop"]})

    if ev_parts:
        ev = pd.concat(ev_parts, ignore_index=True)
        before = len(ev)
        ev = ev.drop_duplicates(subset=["animal", "kind", "event", "time"]).sort_values("time")
        if len(ev) < before:
            out.notes.append(f"{before - len(ev)} duplicate event row(s) removed "
                             "(same animal, event and time in more than one file).")
        out.events = ev.reset_index(drop=True)
    else:
        out.events = pd.DataFrame(columns=["animal", "session", "kind", "event", "time", "fed"])

    cov = pd.DataFrame(cov_rows)
    if not cov.empty and not out.events.empty:
        for i, r in cov.iterrows():
            e = out.events[(out.events["animal"] == r["animal"]) & (out.events["session"] == r["session"])]
            if pd.isna(r["start"]) and len(e):
                cov.at[i, "start"] = e["time"].min()
            if pd.isna(r["stop"]) and len(e):
                cov.at[i, "stop"] = e["time"].max()

    # ---- FED health: status rows from the data files + connect/disconnect/reboot
    parts = list(hl_parts)
    if not out.events.empty and "kind" in out.events:
        h = out.events[out.events["kind"] == "health"]
        if len(h):
            parts.append(pd.DataFrame({"animal": h["animal"].to_numpy(), "session": h["session"].to_numpy(),
                                       "event": h["event"].to_numpy(), "time": h["time"].to_numpy(),
                                       "detail": "", "source": "data",
                                       "poke_before_s": _poke_before(out.events, h)}))
    out.health = (pd.concat(parts, ignore_index=True).sort_values("time").reset_index(drop=True)
                  if parts else pd.DataFrame(columns=["animal", "session", "event", "time", "detail", "source"]))
    out.alerts = (pd.concat(al_parts, ignore_index=True).sort_values("time").reset_index(drop=True)
                  if al_parts else pd.DataFrame(columns=["animal", "fed", "session", "time", "rule", "message"]))

    # USB disconnects are not logged time: cut them out of the coverage
    if not cov.empty and not out.health.empty:
        cov = _cut_disconnects(cov, out)
    out.coverage = cov
    out.sessions = pd.DataFrame(sess_rows).sort_values("start").reset_index(drop=True) \
        if sess_rows else pd.DataFrame()
    if not out.events.empty:
        n = (out.events["kind"] == "pellet").sum()
        out.notes.append(f"Read {n} pellets for {len(out.animals)} animal(s) "
                         f"({', '.join(out.animals)}) from {len(groups)} session(s).")
        nh = int((out.events["kind"] == "health").sum())
        if nh:
            out.notes.append(f"{nh} FED status row(s) (jams, blocked pokes, SD card, restarts) - see "
                             "the 'FED health' view. They are not counted as pokes or pellets.")
    if len(out.alerts):
        out.notes.append(f"{len(out.alerts)} email alert(s) read from alerts_log.csv.")
    return out


def _poke_before(events, h):
    """For LeftBlocked/RightBlocked rows: Poke_Time of the poke row logged just before
    (the beam had already been broken that long when the 'Blocked' row was written)."""
    res = np.full(len(h), np.nan)
    if "poke_time_s" not in events:
        return res
    pk = events[events["kind"] == "poke"]
    for i, (idx, r) in enumerate(h.iterrows()):
        if r["event"] not in ("LeftBlocked", "RightBlocked"):
            continue
        side = r["event"][:-7]
        c = pk[(pk["animal"] == r["animal"]) & (pk["event"].str.startswith(side))
               & (pk["time"] <= r["time"]) & (pk["time"] >= r["time"] - pd.Timedelta(seconds=5))]
        if len(c):
            res[i] = float(c["poke_time_s"].iloc[-1])
    return res


def _cut_disconnects(cov, out):
    rows = []
    for _, r in cov.iterrows():
        segs = [(r["start"], r["stop"])]
        if pd.notna(r["start"]) and pd.notna(r["stop"]):
            iv = health_intervals(out, r["animal"], [r["session"]], kinds=("disconnected",),
                                  _coverage=pd.DataFrame([r]))
            for _, d in iv.iterrows():
                nxt = []
                for a, b in segs:
                    if d["stop"] <= a or d["start"] >= b:
                        nxt.append((a, b))
                        continue
                    if d["start"] > a:
                        nxt.append((a, d["start"]))
                    if d["stop"] < b:
                        nxt.append((d["stop"], b))
                segs = nxt
        for a, b in segs:
            rows.append({**r.to_dict(), "start": a, "stop": b})
    return pd.DataFrame(rows, columns=cov.columns)


# =============================================================================
# FED HEALTH
# =============================================================================

def health_intervals(fed, animal, sessions=None, kinds=None, _coverage=None):
    """
    Periods when something was wrong with one animal's FED.
    -> DataFrame kind, label, start, stop, duration_s, session, open (True if
       the period had not ended when the session's logging stopped).

    kinds: jam (from JamClearing or Jammed until a pellet drops / JamCleared /
           FeedAborted / restart), jammed (from Jammed until the same end),
           disconnected (DISCONNECTED -> CONNECTED), blocked_left /
           blocked_right (beam broken; starts Poke_Time before the Blocked row),
           sd_error (SDError -> SDRecovered).
    """
    cols = ["kind", "label", "start", "stop", "duration_s", "session", "open"]
    h = fed.health
    if h is None or h.empty:
        return pd.DataFrame(columns=cols)
    m = h["animal"] == animal
    if sessions is not None:
        m &= h["session"].isin(list(sessions))
    h = h[m].sort_values("time")
    if h.empty:
        return pd.DataFrame(columns=cols)
    cov = _coverage if _coverage is not None else fed.coverage_of(animal, sessions)
    ev = fed.events
    rows = []
    for session, hs in h.groupby("session", sort=False):
        c = cov[cov["session"] == session] if len(cov) else cov
        s_end = c["stop"].max() if len(c) and pd.notna(c["stop"].max()) else hs["time"].max()
        pel = ev[(ev["animal"] == animal) & (ev["session"] == session)
                 & ev["kind"].isin(["pellet", "dispensed"])]["time"].sort_values() if len(ev) else pd.Series([], dtype="datetime64[ns]")
        opened = {}

        def close(k, t, rows=rows, session=session):
            if k in opened:
                a = opened.pop(k)
                rows.append({"kind": k, "start": a, "stop": max(t, a), "session": session, "open": False})

        for r in hs.itertuples():
            e, t = str(r.event), r.time
            if pd.isna(t):
                continue
            # a pellet (taken or dropped) since the jam started ends it
            for k in ("jam", "jammed"):
                if k in opened:
                    p = pel[(pel > opened[k]) & (pel <= t)]
                    if len(p):
                        close(k, p.iloc[0])
            if e == "JamClearing":
                opened.setdefault("jam", t)
            elif e == "Jammed":
                opened.setdefault("jam", t)
                opened.setdefault("jammed", t)
            elif e in ("JamCleared", "FeedAborted") or e.startswith("Restart_") or e == "FED_REBOOT":
                close("jammed", t)
                close("jam", t)
            if e in ("DISCONNECTED",):
                opened.setdefault("disconnected", t)
            elif e in ("CONNECTED", "RECONNECTED"):
                close("disconnected", t)
            elif e in ("LeftBlocked", "RightBlocked"):
                k = "blocked_" + e[:-7].lower()
                back = getattr(r, "poke_before_s", np.nan)
                back = float(back) if pd.notna(back) and back > 0 else 30.0
                opened.setdefault(k, t - pd.Timedelta(seconds=back))
            elif e in ("LeftUnblocked", "RightUnblocked"):
                close("blocked_" + e[:-9].lower(), t)
            elif e == "SDError":
                opened.setdefault("sd_error", t)
            elif e == "SDRecovered":
                close("sd_error", t)
            elif e == "RECORDING_STOP":
                for k in list(opened):
                    close(k, t)
        for k in ("jam", "jammed"):            # jams ended by a pellet after the last status row
            if k in opened:
                p = pel[pel > opened[k]]
                if len(p):
                    close(k, p.iloc[0])
        for k, a in list(opened.items()):
            end = s_end if pd.notna(s_end) and s_end > a else hs["time"].max()
            rows.append({"kind": k, "start": a, "stop": max(end, a), "session": session, "open": True})
    out = pd.DataFrame(rows, columns=["kind", "start", "stop", "session", "open"])
    if kinds is not None:
        out = out[out["kind"].isin(list(kinds))]
    if out.empty:
        return pd.DataFrame(columns=cols)
    out = out.assign(start=pd.to_datetime(out["start"]), stop=pd.to_datetime(out["stop"]))
    out["label"] = out["kind"].map(HEALTH_LABELS)
    out["duration_s"] = (out["stop"] - out["start"]).dt.total_seconds()
    return out.sort_values("start").reset_index(drop=True)[cols]


def health_points(fed, animal, sessions=None):
    """One-off FED events: restarts (with cause), aborted dispenses, controller errors,
    email alerts. -> DataFrame time, kind, label, detail."""
    rows = []
    h = fed.health
    if h is not None and not h.empty:
        m = h["animal"] == animal
        if sessions is not None:
            m &= h["session"].isin(list(sessions))
        for r in h[m].itertuples():
            e = str(r.event)
            if e.startswith("Restart_"):
                cause = e[len("Restart_"):]
                rows.append({"time": r.time, "kind": "restart", "label": f"Restart ({cause})",
                             "detail": RESTART_WHY.get(cause, cause)})
            elif e == "FED_REBOOT":
                rows.append({"time": r.time, "kind": "reboot_seen", "label": "Restart seen by the PC",
                             "detail": str(r.detail)})
            elif e == "FeedAborted":
                rows.append({"time": r.time, "kind": "aborted", "label": "Dispense aborted",
                             "detail": "stuck dispense cancelled from the controller"})
            elif e == "FED_ERROR":
                rows.append({"time": r.time, "kind": "error", "label": "FED error message",
                             "detail": str(r.detail)})
    a = fed.alerts
    if a is not None and not a.empty:
        m = a["animal"] == str(animal).lower()
        if sessions is not None:
            m &= a["session"].isin(list(sessions))
        for r in a[m].itertuples():
            ok = str(r.rule).endswith("_ok")
            rows.append({"time": r.time, "kind": "alert_ok" if ok else "alert",
                         "label": ("OK email: " if ok else "Email alert: ") + str(r.message),
                         "detail": str(r.rule)})
    out = pd.DataFrame(rows, columns=["time", "kind", "label", "detail"])
    return out.sort_values("time").reset_index(drop=True)


def health_summary(fed, sessions=None, animals=None):
    """One row per animal: counts and total durations of every FED problem."""
    rows = []
    for an in (animals or fed.animals):
        iv = health_intervals(fed, an, sessions)
        pts = health_points(fed, an, sessions)
        ev = fed.events
        m = (ev["animal"] == an) & ((ev["session"].isin(list(sessions))) if sessions is not None else True)

        def n(k):
            return int((iv["kind"] == k).sum())

        def mins(k):
            return round(float(iv.loc[iv["kind"] == k, "duration_s"].sum()) / 60.0, 1)
        restarts = pts[pts["kind"] == "restart"]
        causes = restarts["label"].str.extract(r"\((.*)\)")[0].value_counts() if len(restarts) else pd.Series(dtype=int)
        fb = int((ev.loc[m, "time_source"] == "PC_Received").sum()) if "time_source" in ev else 0
        rows.append({
            "animal": an,
            "jams": n("jam"),
            "jam time (min)": mins("jam"),
            "longest jam (min)": round(float(iv.loc[iv["kind"] == "jam", "duration_s"].max()) / 60.0, 1) if n("jam") else 0.0,
            "JAMMED": n("jammed"),
            "dispenses aborted": int((pts["kind"] == "aborted").sum()),
            "disconnects": n("disconnected"),
            "disconnected (min)": mins("disconnected"),
            "restarts": len(restarts),
            "restart causes": ", ".join(f"{k} ×{v}" for k, v in causes.items()) or "—",
            "pokes blocked": n("blocked_left") + n("blocked_right"),
            "blocked (min)": round(mins("blocked_left") + mins("blocked_right"), 1),
            "SD errors": n("sd_error"),
            "email alerts": int((pts["kind"] == "alert").sum()),
            "rows timed by PC receive": fb,
        })
    return pd.DataFrame(rows)


# =============================================================================
# ANALYSIS
# =============================================================================

def to_seconds(times, clock_start):
    """Local Timestamps -> seconds since a recording's clock start."""
    if times is None or len(times) == 0:
        return np.array([], dtype=float)
    return (pd.DatetimeIndex(times) - pd.Timestamp(clock_start)).total_seconds().to_numpy(float)


def intake_bins(event_sec, t0, t1, bin_sec, coverage_sec=None):
    """
    Pellets per bin on [t0, t1). Bins with no FED logging at all are NaN, not
    0: "the FED was not recording" is not "the animal did not eat". A bin is
    kept when any part of it was covered.
    Returns DataFrame: start, mid, pellets, covered_frac.
    """
    edges = np.arange(t0, t1 + bin_sec, bin_sec, dtype=float)
    if edges.size < 2:
        edges = np.array([t0, t0 + bin_sec])
    counts, _ = np.histogram(np.asarray(event_sec, float), bins=edges)
    counts = counts.astype(float)
    cov = np.ones(counts.size)
    if coverage_sec is not None:
        cov = np.zeros(counts.size)
        for a, b in coverage_sec:
            lo = np.clip(a, edges[:-1], edges[1:])
            hi = np.clip(b, edges[:-1], edges[1:])
            cov += np.maximum(hi - lo, 0)
        cov = np.clip(cov / bin_sec, 0, 1)
        counts[cov <= 0] = np.nan
    return pd.DataFrame({"start": edges[:-1], "mid": edges[:-1] + bin_sec / 2,
                         "pellets": counts, "covered_frac": cov})


def find_meals(event_sec, max_gap_sec=60.0, min_pellets=2):
    """
    Group pellets into meals: consecutive pellets less than max_gap_sec apart
    belong to one meal; meals with fewer than min_pellets are dropped.
    Returns DataFrame: meal, onset, offset, n_pellets, duration_s, gap_before_s.
    """
    t = np.sort(np.asarray(event_sec, float))
    cols = ["meal", "onset", "offset", "n_pellets", "duration_s", "gap_before_s"]
    if t.size == 0:
        return pd.DataFrame(columns=cols)
    br = np.flatnonzero(np.diff(t) > max_gap_sec)
    starts = np.r_[0, br + 1]
    ends = np.r_[br, t.size - 1]
    rows, prev = [], None
    for s, e in zip(starts, ends):
        n = int(e - s + 1)
        if n < min_pellets:
            continue
        rows.append({"onset": t[s], "offset": t[e], "n_pellets": n, "duration_s": t[e] - t[s],
                     "gap_before_s": np.nan if prev is None else t[s] - prev})
        prev = t[e]
    out = pd.DataFrame(rows, columns=cols[1:])
    out.insert(0, "meal", np.arange(1, len(out) + 1))
    return out


def rolling_rate(event_sec, grid_sec, window_sec=1800.0):
    """Pellets per hour in a centred window, on a time grid."""
    t = np.sort(np.asarray(event_sec, float))
    g = np.asarray(grid_sec, float)
    if t.size == 0:
        return np.zeros_like(g)
    h = window_sec / 2.0
    n = np.searchsorted(t, g + h, side="right") - np.searchsorted(t, g - h, side="left")
    return n * 3600.0 / window_sec


def window_means(t, y, centres, a, b):
    """Mean of y over [c+a, c+b] for every centre c (NaN-aware, vectorised)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    c = np.asarray(centres, float)
    ok = np.isfinite(y)
    cs = np.r_[0.0, np.cumsum(np.where(ok, y, 0.0))]
    cn = np.r_[0, np.cumsum(ok)]
    i0 = np.searchsorted(t, c + a, side="left")
    i1 = np.searchsorted(t, c + b, side="right")
    n = cn[i1] - cn[i0]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (cs[i1] - cs[i0]) / n
    m[n == 0] = np.nan
    inside = (c + a >= t[0]) & (c + b <= t[-1])
    m[~inside] = np.nan
    return m


def aligned_matrix(t, y, events, pre, post, baseline=None, dt=None):
    """
    Trials x time matrix of y around each event, on a regular grid from -pre
    to +post. With baseline=(a, b) each trial has its own baseline mean
    subtracted. Trials whose window leaves the recording are dropped.
    Returns (matrix, taxis, kept_events).
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ev = np.asarray(events, float)
    if t.size < 2 or ev.size == 0:
        return np.zeros((0, 0)), np.zeros(0), np.zeros(0)
    dt = dt or float(np.median(np.diff(t)))
    tax = np.arange(-pre, post + dt / 2, dt)
    keep = ev[(ev - pre >= t[0]) & (ev + post <= t[-1])]
    if keep.size == 0:
        return np.zeros((0, tax.size)), tax, keep
    good = np.isfinite(y)
    rows = np.vstack([np.interp(e + tax, t[good], y[good]) for e in keep])
    # Blank samples that fall in NaN stretches of the original trace.
    nan_t = t[~good]
    if nan_t.size:
        for r, e in enumerate(keep):
            near = np.searchsorted(nan_t, e + tax)
            hit = np.zeros(tax.size, bool)
            for k in (near - 1, near):
                k = np.clip(k, 0, nan_t.size - 1)
                hit |= np.abs(nan_t[k] - (e + tax)) < dt / 2
            rows[r, hit] = np.nan
    if baseline is not None:
        bm = (tax >= baseline[0]) & (tax <= baseline[1])
        rows = rows - np.nanmean(rows[:, bm], axis=1, keepdims=True)
    return rows, tax, keep


def permutation_test(t, y, events, base, resp, allowed_sec=None, n_perm=2000,
                     exclude_sec=None, seed=0):
    """
    Does the signal change around eating more than around random times?

      delta(event) = mean(y over resp window) - mean(y over base window)

    The observed mean delta across events is compared with the same statistic
    for equally many random times, drawn from `allowed_sec` intervals (times
    when the FED was logging and photometry was recording), avoiding
    +/- exclude_sec around real events. Two-sided p = share of random means at
    least as far from the random-mean centre as the observed one.
    Returns dict: n, observed, deltas, null_mean, null_sd, p, ci95.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    ev = np.asarray(events, float)
    d = window_means(t, y, ev, *resp) - window_means(t, y, ev, *base)
    d = d[np.isfinite(d)]
    res = {"n": int(d.size), "observed": float(np.mean(d)) if d.size else np.nan, "deltas": d,
           "null_mean": np.nan, "null_sd": np.nan, "p": np.nan, "ci95": (np.nan, np.nan)}
    if d.size < 3:
        return res
    lo_pad, hi_pad = min(base[0], resp[0]), max(base[1], resp[1])
    iv = allowed_sec if allowed_sec is not None and len(allowed_sec) else [(t[0], t[-1])]
    iv = [(max(a, t[0]) - lo_pad, min(b, t[-1]) - hi_pad) for a, b in iv]
    iv = [(a, b) for a, b in iv if b > a]
    if not iv:
        return res
    lens = np.array([b - a for a, b in iv])
    excl = exclude_sec if exclude_sec is not None else (hi_pad - lo_pad)
    ev_sorted = np.sort(ev)
    pool = []
    need = d.size * n_perm
    for _ in range(20):                      # draw in batches until enough
        k = int(need * 1.5) + 100
        which = rng.choice(len(iv), size=k, p=lens / lens.sum())
        cand = np.array([iv[w][0] for w in which]) + rng.random(k) * lens[which]
        j = np.searchsorted(ev_sorted, cand)
        dist = np.minimum(np.abs(cand - ev_sorted[np.clip(j - 1, 0, ev_sorted.size - 1)]),
                          np.abs(cand - ev_sorted[np.clip(j, 0, ev_sorted.size - 1)]))
        cand = cand[dist > excl]
        dd = window_means(t, y, cand, *resp) - window_means(t, y, cand, *base)
        pool.append(dd[np.isfinite(dd)])
        if sum(p.size for p in pool) >= need:
            break
    pool = np.concatenate(pool)
    if pool.size < need:
        n_perm = max(1, pool.size // d.size)
    null = pool[: n_perm * d.size].reshape(n_perm, d.size).mean(axis=1)
    centre = float(np.mean(null))
    res.update(null_mean=centre, null_sd=float(np.std(null, ddof=1)) if n_perm > 1 else np.nan,
               p=float((np.sum(np.abs(null - centre) >= abs(res["observed"] - centre)) + 1) / (n_perm + 1)),
               ci95=tuple(np.percentile(null, [2.5, 97.5])))
    return res
