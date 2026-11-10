"""Tests for fed3: reading FED3 controller output and aligning it to photometry."""
import glob
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import fed3

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


HDR = ("MM:DD:YYYY hh:mm:ss,Library_Version,Session_type,Device_Number,Battery_Voltage,Motor_Turns,FR,"
       "Event,Active_Poke,Left_Poke_Count,Right_Poke_Count,Pellet_Count,Block_Pellet_Count,Retrieval_Time,"
       "InterPelletInterval,Poke_Time,PC_Timestamp,PC_Received,PC_Received_Unix")


def row(ts, event="Pellet", count=1, retr="2.1"):
    t = pd.Timestamp(ts)
    return (f"{t:%m/%d/%Y %H:%M:%S},1.17.3,PC_Free,0,4.2,1,0,{event},Left,0,0,{count},0,{retr},nan,nan,"
            f"{t:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{t:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{t.timestamp():.3f}")


tmp = Path(tempfile.mkdtemp())
s1 = tmp / "FED3_data" / "longi_20261015_103500"
s2 = tmp / "FED3_data" / "test_20261015_160133"
s3 = tmp / "FED3_data" / "refeed_20261015_193635"
for s in (s1, s2, s3):
    s.mkdir(parents=True)

# Session 1: m10 eats 3 pellets; the counter restarts after a NEWFILE (old count 5 first).
(s1 / "FED001_m10_data.csv").write_text("\n".join([HDR,
    row("2026-10-15 11:49:10.052", count=5, retr="Timed_out"),
    row("2026-10-15 11:49:11.995", count=1), row("2026-10-15 11:49:31.517", count=2),
    row("2026-10-15 11:52:00.000", event="LeftWithPellet")]) + "\n")
# m16 written by two FEDs: one pc_events line cut off by the other writer.
(s1 / "FED000_m16_data.csv").write_text("\n".join([HDR, row("2026-10-15 10:36:00.000", count=1)]) + "\n")
(s1 / "FED000_m16_pc_events.csv").write_text(
    "PC_Time,PC_Time_Unix,Event,Detail,FED_PC_Timestamp\n"
    "2026-10-15 10:35:00.215,1792078500.215,RECORDING_START,port=COM16;animal=m16,\n"
    "2026-10-15 10:35:00.650,1792078500.650,PELLET_DISPENSED,fed_millis=442026-10-15 13:40:02.594,1792089602.595,RECORDING_STOP,,\n"
    "792078500.774,SD_NEWFILE,\"NEWFILE,queued\",\n"
    "2026-10-15 10:35:58.360,1792078558.360,PELLET_DISPENSED,fed_millis=44406059,2026-10-15 10:35:58.359\n"
    "2026-10-15 13:40:02.594,1792089602.595,RECORDING_STOP,,\n")
(s1 / "session_info.json").write_text(json.dumps({
    "session_name": "longi", "start_pc_time": "2026-10-15 10:35:00.211",
    "stop_pc_time": "2026-10-15 13:40:02.594", "utc_offset_hours": -5.0,
    "feds": {"COM10": {"fed_number": 0, "animal_id": "m16", "info": {}, "last_status": {}},
             "COM16": {"fed_number": 0, "animal_id": "m16", "info": {}, "last_status": {}},
             "COM27": {"fed_number": 1, "animal_id": "m10", "info": {}, "last_status": {}}}}))
(s2 / "FED001_m10_data.csv").write_text("\n".join([HDR, row("2026-10-15 16:02:16.662")]) + "\n")
(s3 / "FED001_m10_data.csv").write_text("\n".join([HDR, row("2026-10-15 19:40:37.804"),
                                                  row("2026-10-15 19:40:37.804")]) + "\n")   # duplicate
(s3 / "FED000_NA_data.csv").write_text(HDR + "\n")
zp = tmp / "FED3_data.zip"
with zipfile.ZipFile(zp, "w") as z:
    for f in (tmp / "FED3_data").rglob("*"):
        if f.is_file():
            z.write(f, f.relative_to(tmp))
    z.writestr("../../evil.csv", "x")                      # must not escape the work folder

print("=== reading ===")
work = tmp / "work"
fd = fed3.load_fed([zp], work)
check("animals by ID, NA skipped", fd.animals == ["m10", "m16"], str(fd.animals))
check("pellets counted as events, not from Pellet_Count",
      len(fd.times("m10", "pellet", ["longi_20261015_103500"])) == 3)
check("first row after NEWFILE (old count, Timed_out) kept as a real pellet",
      fd.times("m10")[0] == pd.Timestamp("2026-10-15 11:49:10.052"))
check("pokes separate from pellets", (fd.events["kind"] == "poke").sum() == 1)
check("duplicate rows removed", len(fd.times("m10", "pellet", ["refeed_20261015_193635"])) == 1)
check("shared-label warning", any("COM10, COM16" in w for w in fd.warnings))
check("damaged pc_events line reported", any("damaged line" in w for w in fd.warnings))
d16 = fd.events[(fd.events["animal"] == "m16") & (fd.events["kind"] == "dispensed")]
check("cut-off dispense line recovered from its time stamp", len(d16) == 2, str(len(d16)))
check("zip cannot write outside the work folder", not (tmp / "evil.csv").exists()
      and not any(p.name == "evil.csv" and work not in p.parents for p in tmp.rglob("evil.csv")))
cov = fd.coverage_of("m16")
check("coverage from RECORDING_START/STOP", cov["start"].iloc[0] == pd.Timestamp("2026-10-15 10:35:00.215")
      and cov["stop"].iloc[0] == pd.Timestamp("2026-10-15 13:40:02.594"))
check("session table", set(fd.sessions["label"]) == {"longi", "test", "refeed"})
check("animal_from_name", fed3.animal_from_name("m10_fasted_feds-2026-10-15-103416") == "m10"
      and fed3.animal_from_name("FED000_M16_data") == "m16")

print("\n=== analysis ===")
cs = pd.Timestamp("2026-10-15 10:34:16.932")
sec = fed3.to_seconds(fd.times("m10", "pellet", ["longi_20261015_103500"]), cs)
check("to_seconds", abs(sec[0] - (pd.Timestamp("2026-10-15 11:49:10.052") - cs).total_seconds()) < 1e-6)
b = fed3.intake_bins([100, 110, 700], 0, 1800, 600, coverage_sec=[(0, 1200)])
check("bins count pellets", list(b["pellets"][:2]) == [2, 1])
check("unlogged bin is NaN, not 0", np.isnan(b["pellets"].iloc[2]))
m = fed3.find_meals([0, 20, 50, 400, 410, 2000], max_gap_sec=60, min_pellets=2)
check("meals", list(m["n_pellets"]) == [3, 2] and list(m["onset"]) == [0, 400], m.to_string())
r = fed3.rolling_rate([0, 10, 20], np.array([10.0]), 60)
check("rolling rate", abs(r[0] - 180.0) < 1e-9, str(r))

# A signal that rises by 1 for 60 s after every meal onset, plus noise.
rng = np.random.default_rng(1)
t = np.arange(0, 86_400, 1.0)
y = 0.3 * rng.standard_normal(t.size)
onsets = np.arange(3000, 80_000, 2500.0)
for o in onsets:
    y[(t >= o) & (t < o + 60)] += 1.0
wm = fed3.window_means(t, y, onsets, 0, 50) - fed3.window_means(t, y, onsets, -100, -50)
check("window means see the step", abs(np.mean(wm) - 1.0) < 0.1, f"{np.mean(wm):.3f}")
st = fed3.permutation_test(t, y, onsets, (-100, -50), (0, 50), n_perm=500)
check("real response detected (p < 0.01)", st["p"] < 0.01 and abs(st["observed"] - 1) < 0.1,
      f"obs {st['observed']:.3f} p {st['p']:.4f}")
fake = onsets + 1250
y2 = 0.3 * rng.standard_normal(t.size)
st2 = fed3.permutation_test(t, y2, fake, (-100, -50), (0, 50), n_perm=500)
check("no response -> not significant", st2["p"] > 0.05, f"p {st2['p']:.3f}")
mat, tax, kept = fed3.aligned_matrix(t, y, onsets, 120, 240, baseline=(-100, -50))
check("aligned matrix shape", mat.shape == (onsets.size, tax.size))
check("aligned mean ~1 after onset", abs(np.nanmean(mat[:, (tax > 5) & (tax < 50)]) - 1) < 0.1)
yn = y.copy(); yn[5000:5100] = np.nan
mat2, tax2, _ = fed3.aligned_matrix(t, yn, [5050.0], 120, 240)
check("NaN stretches stay NaN in aligned trials", np.isnan(mat2[0, (tax2 > -40) & (tax2 < 40)]).any())

print("\n=== real data (if present) ===")
real = glob.glob("/root/.claude/uploads/*/*FED3_data.zip") + glob.glob("FED3_data.zip")
if real:
    fr = fed3.load_fed([real[0]], tempfile.mkdtemp())
    check("4 animals", fr.animals == ["m10", "m11", "m14", "m16"], str(fr.animals))
    n = fr.events[fr.events["kind"] == "pellet"].groupby("animal").size().to_dict()
    check("pellet totals", n == {"m10": 210, "m11": 186, "m14": 170, "m16": 186}, str(n))
else:
    print("  (skipped)")

print("\n=== recording -> FED animal ===")
A = ["k_m2", "k_m5", "k_m7", "k_m8", "k_m11", "m1"]
check("prefixed ID found in recording name", fed3.guess_animal("k_M5_longitudinal-2026-11-04-155731", A) == "k_m5")
check("m1 does not take m11", fed3.guess_animal("k_M11_day2", A) == "k_m11")
check("plain ID still works", fed3.guess_animal("m1_fasted-2026", A) == "m1")
check("no match -> None", fed3.guess_animal("rat3_session", A) is None)

print("\n=== FED health (controller 2.0 / firmware PCCtrl-2.0) ===")
HT0 = pd.Timestamp("2026-11-10 09:00:00")
hs = tmp / "H" / "health_20261110_090000"
hs.mkdir(parents=True)


def hrow(sec, event, L=0, P=0, poke="nan", nan_ts=False, v="4.15"):
    t = HT0 + pd.Timedelta(seconds=sec)
    rec = t + pd.Timedelta(milliseconds=2)
    ts = "NaN" if nan_ts else f"{t:%Y-%m-%d %H:%M:%S.%f}"[:-3]
    return (f"{t:%m/%d/%Y %H:%M:%S},1.17.3,PC_Free,3,{v},nan,0,{event},Left,{L},0,{P},0,nan,nan,{poke},"
            f"{ts},{rec:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{rec.timestamp():.3f}")


def pcl(sec, ev, det=""):
    t = HT0 + pd.Timedelta(seconds=sec)
    return f"{t:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{t.timestamp():.3f},{ev},{det},"


data = [hrow(30, "Pellet", P=1), hrow(600, "JamClearing", P=1), hrow(900, "JamCleared", P=1),
        hrow(905, "Pellet", P=2), hrow(1200, "JamClearing", P=2), hrow(1800, "Jammed", P=2),
        hrow(2400, "Left", L=1, P=2, poke="30.0"), hrow(2400.5, "LeftBlocked", L=1, P=2),
        hrow(3000, "LeftUnblocked", L=1, P=2), hrow(3300, "JamCleared", P=2), hrow(3305, "Pellet", P=3),
        hrow(4000, "Restart_Watchdog", L=1, P=3, nan_ts=True), hrow(4002, "Left", L=2, P=3, poke="0.2", nan_ts=True),
        hrow(5000, "SDError", L=2, P=3), hrow(5600, "SDRecovered", L=2, P=3), hrow(6000, "FeedAborted", L=2, P=3)]
(hs / "FED003_m20_data.csv").write_text(HDR + "\n" + "\n".join(data) + "\n")
(hs / "FED003_m20_pc_events.csv").write_text(
    "PC_Time,PC_Time_Unix,Event,Detail,FED_PC_Timestamp\n" + "\n".join([
        pcl(0, "RECORDING_START", "port=COM5"), pcl(0, "CONNECTED", "COM5"), pcl(3999, "DISCONNECTED", "COM5"),
        pcl(4003, "CONNECTED", "COM5"), pcl(4004, "FED_REBOOT", "reset=Watchdog"), pcl(4100, "TTL_PULSE", "x"),
        pcl(4200, "FED_ERROR", "unknown command"), pcl(7200, "RECORDING_STOP")]) + "\n")
# second FED: port-suffixed file name and a long disconnect
(hs / "FED005_m21_COM7_data.csv").write_text(HDR + "\n" + "\n".join([hrow(100, "Pellet", P=1),
                                                                       hrow(5000, "Pellet", P=2)]) + "\n")
(hs / "FED005_m21_COM7_pc_events.csv").write_text(
    "PC_Time,PC_Time_Unix,Event,Detail,FED_PC_Timestamp\n" + "\n".join([
        pcl(0, "RECORDING_START"), pcl(1000, "DISCONNECTED", "COM7"), pcl(2800, "CONNECTED", "COM7"),
        pcl(7200, "RECORDING_STOP")]) + "\n")
(hs / "alerts_log.csv").write_text(
    "PC_Time,PC_Time_Unix,FED,Animal,Rule,Message,FED_PC_Timestamp\n"
    + f"{HT0 + pd.Timedelta(seconds=1800):%Y-%m-%d %H:%M:%S}.000,0,FED003,m20,jammed,JAMMED,\n"
    + f"{HT0 + pd.Timedelta(seconds=3305):%Y-%m-%d %H:%M:%S}.000,0,FED003,m20,jammed_ok,jam cleared,\n")
fh = fed3.load_fed([hs], tmp / "work_h")
check("health: animals incl. port-suffixed file", fh.animals == ["m20", "m21"], str(fh.animals))
check("health: no false 'damaged line' warnings", not fh.warnings, str(fh.warnings))
e20 = fh.events[fh.events["animal"] == "m20"]
check("health rows are not pokes", int((e20["kind"] == "poke").sum()) == 2,
      str(e20.groupby("kind").size().to_dict()))
check("health rows have kind 'health'", int((e20["kind"] == "health").sum()) == 11)
check("rows without PC_Timestamp kept (PC_Received)", int((e20["time_source"] == "PC_Received").sum()) == 2)
iv = fed3.health_intervals(fh, "m20")
jam = iv[iv["kind"] == "jam"]
check("two jams", len(jam) == 2, str(len(jam)))
check("first jam 600->900 s", abs(jam["duration_s"].iloc[0] - 300) < 1, str(jam["duration_s"].tolist()))
check("second jam 1200->3300 s", abs(jam["duration_s"].iloc[1] - 2100) < 1)
jm = iv[iv["kind"] == "jammed"]
check("JAMMED 1800->3300 s", len(jm) == 1 and abs(jm["duration_s"].iloc[0] - 1500) < 1)
bl = iv[iv["kind"] == "blocked_left"]
check("blocked poke starts Poke_Time before the row", len(bl) == 1 and abs(bl["duration_s"].iloc[0] - 630.0) < 1,
      str(bl["duration_s"].tolist()))
check("SD error 600 s", abs(iv.loc[iv["kind"] == "sd_error", "duration_s"].iloc[0] - 600) < 1)
pts = fed3.health_points(fh, "m20")
check("restart with cause", (pts["label"] == "Restart (Watchdog)").sum() == 1, str(pts["label"].tolist()))
check("alerts read", (pts["kind"] == "alert").sum() == 1 and (pts["kind"] == "alert_ok").sum() == 1)
check("dispense aborted", (pts["kind"] == "aborted").sum() == 1)
cov21 = fh.coverage_of("m21")
check("disconnect cut out of coverage", len(cov21) == 2
      and abs((cov21["stop"] - cov21["start"]).dt.total_seconds().sum() - (7200 - 1800)) < 1,
      str(cov21[["start", "stop"]].values.tolist()))
smm = fed3.health_summary(fh).set_index("animal")
check("summary counts", smm.loc["m20", "jams"] == 2 and smm.loc["m20", "JAMMED"] == 1
      and smm.loc["m21", "disconnects"] == 1 and smm.loc["m21", "disconnected (min)"] == 30.0,
      smm[["jams", "JAMMED", "disconnects", "disconnected (min)"]].to_dict())
check("pellets unchanged", fh.times("m20", "pellet").size == 3 and fh.times("m21", "pellet").size == 2)
check("old data: no health rows", (fr.events["kind"] == "health").sum() == 0 if "fr" in dir() else True)

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
