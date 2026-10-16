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

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
