"""Tests for whole-ZIP intake, the new acquisition build (system-clock stamps, padded
header, snapshots, CSV recordings), baseline normalisation and the longitudinal views.

Run:  python test_intake.py
The fixture writer (make_gp3) is also used to build a demo ZIP:  python test_intake.py --zip out.zip
"""
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import baseline as bl
import fed3
import intake
import longitudinal as lg
import photometry_core as pc

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


VPD = 0.00010071
UTC_OFF = -6 * 3600                      # CST
FED_HDR = ("MM:DD:YYYY hh:mm:ss,Library_Version,Session_type,Device_Number,Battery_Voltage,Motor_Turns,FR,"
           "Event,Active_Poke,Left_Poke_Count,Right_Poke_Count,Pellet_Count,Block_Pellet_Count,Retrieval_Time,"
           "InterPelletInterval,Poke_Time,PC_Timestamp,PC_Received,PC_Received_Unix")


def _hdr_bytes(h, size=4096):
    b = json.dumps(h).encode()
    assert len(b) <= size
    return size.to_bytes(2, "little") + b + b" " * (size - len(b))


def write_ppd(path, t0_unix, n, fs, sig_v, iso_v, board_ppm=30.0, end_time_ok=True, snapshot_of=None,
              pc_stamps=True, seed=0, extra=None):
    """A .ppd exactly as the custom build of 2026-11-16 writes it (4096-byte padded header,
    4 uint16 words + float64 system-clock stamp per sample)."""
    rng = np.random.default_rng(seed)
    tz = timezone(timedelta(seconds=UTC_OFF))
    start = datetime.fromtimestamp(t0_unix, tz).replace(tzinfo=None)
    sys_t = t0_unix + np.arange(n) / fs * (1 - board_ppm * 1e-6)
    stamps = sys_t + rng.uniform(0.0, 0.008, n)
    off = 120
    w = np.empty((n, 4), np.uint16)
    w[:, 0] = ((np.round(sig_v / VPD) + off).astype(np.int64) << 1).astype(np.uint16)
    w[:, 1] = off << 1
    w[:, 2] = ((np.round(iso_v / VPD) + off).astype(np.int64) << 1).astype(np.uint16)
    w[:, 3] = off << 1
    h = {"subject_ID": path.stem.split("-")[0], "date_time": start.isoformat(timespec="milliseconds"),
         "end_time": (start + timedelta(seconds=n / fs + 0.4)).isoformat(timespec="milliseconds")
         if end_time_ok else start.isoformat(timespec="milliseconds"),
         "n_analog_signals": 2, "n_digital_signals": 2, "mode": "2EX_1EM_pulsed", "sampling_rate": fs,
         "volts_per_division": [VPD, VPD], "ADC_max_value": 32768, "LED_current": [20, 20], "version": "1.1",
         "build": "custom 2026-11-16", "delay_period": 300, "LED_on_time_us": 550.0, "LED_duty_percent": 1.1,
         "effective_duty": 2.2}
    if pc_stamps:
        h.update({"pc_timestamps": True, "words_per_sample": 4, "bytes_per_sample": 16,
                  "pc_clock": {"format": "unix_epoch_seconds_float64", "source": "system_clock",
                               "anchor_unix_s": t0_unix, "utc_offset_s": UTC_OFF,
                               "performance_counter_hz": 10000000, "platform": "win32"}})
    if end_time_ok:
        h.update({"n_samples": n, "duration_s": n / fs, "skipped_chunks": 0, "corrupt_chunks": 0,
                  "clock_summary": {"first_sample_unix": float(stamps[0]), "last_sample_unix": float(stamps[-1]),
                                    "board_vs_system_ppm": -board_ppm, "n_clock_steps": 0, "clock_steps": []}})
    if snapshot_of:
        h.update({"snapshot": True, "snapshot_of": snapshot_of,
                  "snapshot_time": (start + timedelta(seconds=n / fs)).isoformat(timespec="seconds")})
    h.update(extra or {})
    if pc_stamps:
        rec = np.empty(n, dtype=[("w", "<u2", (4,)), ("t", "<f8")])
        rec["w"], rec["t"] = w, stamps
        body = rec.tobytes()
    else:
        body = w.tobytes()
    path.write_bytes(_hdr_bytes(h) + body)
    return sys_t


def make_gp3(root, minutes=40.0, fed_extra_min=30.0, fs=20.0, n_mice=3, t0=1794931686.27):
    """Five-mouse style session: PPDs, a snapshot, and a zipped FED3_data folder in one folder."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tz = timezone(timedelta(seconds=UTC_OFF))
    fed_dir = root / "fed_src" / "Gp3_20261117_101500"
    fed_dir.mkdir(parents=True)
    rng = np.random.default_rng(1)
    truth = {}
    for i in range(1, n_mice + 1):
        animal = f"P{i}_84{20 + i}G"
        n = int(minutes * 60 * fs)
        t = np.arange(n) / fs
        pellets = np.sort(rng.uniform(300, (minutes + fed_extra_min) * 60 - 60, 14))
        resp = np.zeros(n)
        for p in pellets:
            resp -= 0.12 * (t >= p) * np.exp(-np.clip(t - p, 0, None) / 120.0)
        bleach = 1 - 0.04 * (1 - np.exp(-t / 900))
        sig = 1.4 * bleach * (1 + resp) * (1 + 0.004 * rng.standard_normal(n))
        iso = 0.52 * bleach * (1 + 0.004 * rng.standard_normal(n))
        start = t0 + 30 * i
        stem = f"Gp3_{animal}-{datetime.fromtimestamp(start, tz):%Y-%m-%d-%H%M%S}"
        write_ppd(root / f"{stem}.ppd", start, n, fs, sig, iso, end_time_ok=(i == n_mice), seed=i)
        if i == 1:
            (root / "snapshots").mkdir(exist_ok=True)
            k = n // 2
            write_ppd(root / "snapshots" / f"{stem}_snap-2026-11-17-103000.ppd", start, k, fs, sig[:k], iso[:k],
                      end_time_ok=False, snapshot_of=f"{stem}.ppd", seed=i)
        rows, ev = [FED_HDR], ["PC_Time,PC_Time_Unix,Event,Detail,FED_PC_Timestamp"]
        pc0 = datetime.fromtimestamp(start - 60, tz).replace(tzinfo=None)
        ev.append(f"{pc0:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{start - 60:.3f},RECORDING_START,port=COM{i};animal="
                  f"{animal},")
        for j, p in enumerate(pellets, 1):
            u = start + p
            ts = datetime.fromtimestamp(u, tz).replace(tzinfo=None)
            dts = datetime.fromtimestamp(u - 4, tz).replace(tzinfo=None)
            rows.append(f"{ts:%m/%d/%Y %H:%M:%S},1.17.3,PC_Free,0,4.1,1,1,Pellet,Left,0,0,{j},{j},4.0,nan,nan,"
                        + f"{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + "," + f"{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{u:.3f}")
            ev.append(f"{dts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{u - 4:.3f},PELLET_DISPENSED,fed_millis=1,"
                      + f"{dts:%Y-%m-%d %H:%M:%S.%f}"[:-3])
        end = start + (minutes + fed_extra_min) * 60
        te = datetime.fromtimestamp(end, tz).replace(tzinfo=None)
        ev.append(f"{te:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{end:.3f},RECORDING_STOP,,")
        (fed_dir / f"FED000_{animal}_data.csv").write_text("\n".join(rows) + "\n")
        (fed_dir / f"FED000_{animal}_pc_events.csv").write_text("\n".join(ev) + "\n")
        (fed_dir / f"FED000_{animal}.xlsx").write_bytes(b"PK not really excel")
        truth[animal] = {"stem": stem, "pellets": pellets, "start": start}
    (fed_dir / "session_info.json").write_text(json.dumps({
        "session_name": "Gp3", "start_pc_time": "2026-11-17 10:15:00.000", "stop_pc_time": None,
        "utc_offset_hours": UTC_OFF / 3600, "feds": {}}))
    fz = root / "FED3_data.zip"
    with zipfile.ZipFile(fz, "w") as z:
        for f in (root / "fed_src").rglob("*"):
            if f.is_file():
                z.write(f, Path("FED3_data") / f.relative_to(root / "fed_src"))
    import shutil
    shutil.rmtree(root / "fed_src")
    (root / "notes.txt").write_text("session notes")
    return truth


def zip_folder(folder, out):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in Path(folder).rglob("*"):
            if f.is_file():
                z.write(f, Path("Gp3_2026-11-17") / f.relative_to(folder))
    return out


if __name__ == "__main__" and "--zip" in sys.argv:
    out = Path(sys.argv[sys.argv.index("--zip") + 1])
    src = Path(tempfile.mkdtemp())
    make_gp3(src, minutes=float(sys.argv[sys.argv.index("--min") + 1]) if "--min" in sys.argv else 40.0)
    zip_folder(src, out)
    print("wrote", out)
    raise SystemExit(0)


tmp = Path(tempfile.mkdtemp())
src = tmp / "src"
truth = make_gp3(src)
outer = zip_folder(src, tmp / "Gp3_all.zip")

print("=== intake: one ZIP with PPDs, a snapshot and a zipped FED3_data folder ===")
files = intake.expand([outer], tmp / "unz")
res = intake.classify(files)
names = sorted(p.name for p in res["photometry"])
check("three finished recordings found", len(names) == 3 and all("_snap-" not in n for n in names), str(names))
check("snapshot skipped because the finished recording is there",
      any("_snap-" in n and "finished recording" in why for n, why in res["skipped"]))
fed_names = sorted(p.name for p in res["fed"])
check("FED data and pc_events CSVs and session_info.json routed to FED",
      sum(n.endswith("_data.csv") for n in fed_names) == 3 and sum(n.endswith("_pc_events.csv") for n in fed_names)
      == 3 and "session_info.json" in fed_names, str(fed_names))
check("FED CSVs never reach the photometry CSV reader", not any("FED000" in p.name for p in res["csv"]))
check("Excel copies and notes skipped with a reason",
      any(n.endswith(".xlsx") and "Excel" in why for n, why in res["skipped"]) and
      any(n == "notes.txt" for n, _ in res["skipped"]))
fd = fed3.load_fed([str(p) for p in res["fed"]], tmp / "fedwork")
check("FED animals read from the nested ZIP", sorted(fd.animals) == sorted(a.lower() for a in truth), str(fd.animals))
pair = intake.pairing(res["photometry"], fd.animals)
check("recordings paired with FED animals by name",
      all(pair[v["stem"]] == a.lower() for a, v in truth.items()), str(pair))
check("FED CSV is classified as FED even when renamed", intake.csv_kind(
    next(p for p in res["fed"] if p.name.endswith("_data.csv")).rename(tmp / "renamed.csv")) == "fed")

print("\n=== snapshot used when the finished recording is missing ===")
only_snap = [p for p in files if "_snap-" in p.name or p.suffix == ".csv"]
r2 = intake.classify(only_snap)
check("newest snapshot used", len(r2["photometry"]) == 1 and "_snap-" in r2["photometry"][0].name)

print("\n=== the new acquisition build ===")
ppd = next(p for p in res["photometry"] if "P1_" in p.name)
d = pc.read_ppd(ppd)
clk = d["clock"]
fs = 20.0
n_true = int(40 * 60 * fs)
check("16-byte samples read: duration not doubled", d["n_samples"] == n_true and abs(d["duration_sec"] - 2400) < 1,
      f"{d['n_samples']} samples, {d['duration_sec']:.1f} s")
check("channels in volts as written (465 ~1.33 V after the dips, 405 ~0.51 V)",
      abs(np.median(d["analog_1"]) - 1.33) < 0.04 and abs(np.median(d["analog_2"]) - 0.505) < 0.02,
      f"{np.median(d['analog_1']):.3f} / {np.median(d['analog_2']):.3f}")
check("system-clock stamps: time axis follows the PC clock", clk["source"] == "pc" and clk["system_clock"])
sys_t = truth["P1_8421G"]["start"] + np.arange(n_true) / fs * (1 - 30e-6)
tt = pc.ppd_time_axis(clk, n_true)
err = (clk["start_unix"] + tt) - sys_t
check("every sample within 10 ms of its true system-clock time", np.max(np.abs(err)) < 0.010,
      f"max {np.max(np.abs(err)) * 1e3:.1f} ms")
d_s = pc.read_ppd(ppd, clock_source="sample")
err_s = (d_s["clock"]["start_unix"] + pc.ppd_time_axis(d_s["clock"], n_true)) - sys_t
check("...while the board's sample clock would drift (30 ppm)", np.max(np.abs(err_s)) > 0.05,
      f"max {np.max(np.abs(err_s)) * 1e3:.0f} ms")
check("end_time equal to the start is explained, not a warning",
      any("never updated" in n for n in clk["notes"]) and not any("end_time" in w for w in clk["warnings"]))
check("LED on-time reported", any("LED pulse: 0.55 ms" in n for n in clk["notes"]))
check("start time in local time (UTC-6)", str(clk["start_local"]).startswith("2026-11-17T10:08:36"),
      str(clk["start_local"]))
p3 = next(p for p in res["photometry"] if "P3_" in p.name)
c3 = pc.read_ppd(p3)["clock"]
check("proper end_time passes the cross-check", c3["end_time_check"] and
      -0.5 <= c3["end_time_check"]["end_minus_last_sample_sec"] <= 10, str(c3["end_time_check"]))
snap = next(p for p in files if "_snap-" in p.name)
check("snapshot is labelled", any("SNAPSHOT" in n for n in pc.read_ppd(snap)["clock"]["notes"]))

# a clock step listed in the header is not lost samples
stp = tmp / "step.ppd"
n = 20 * 600
tsys = write_ppd(stp, 1794931686.0, n, 20.0, np.full(n, 1.2), np.full(n, 0.5), end_time_ok=True)
raw = stp.read_bytes()
hl = int.from_bytes(raw[:2], "little")
rec = np.frombuffer(raw[2 + hl:], dtype=[("w", "<u2", (4,)), ("t", "<f8")]).copy()
rec["t"][6000:] += 1.5
h = pc.parse_header_bytes(raw[2:2 + hl])
h["clock_summary"]["clock_steps"] = [{"sample": 6000, "unix": float(rec["t"][6000]), "step_s": 1.5}]
h["clock_summary"]["n_clock_steps"] = 1
stp.write_bytes(_hdr_bytes(h) + rec.tobytes())
cs_ = pc.read_ppd(stp)["clock"]
check("Windows clock step: recognised, not reported as lost samples",
      len(cs_["clock_steps"]) == 1 and not cs_["gaps"] and abs(cs_["clock_steps"][0]["step_sec"] - 1.5) < 0.1,
      str(cs_["clock_steps"]) + str(cs_["gaps"]))
cs_s = pc.read_ppd(stp, clock_source="sample")["clock"]
check("...and the sample-clock axis gets no gap for it", abs(cs_s["duration_sec"] - 600) < 0.5,
      f"{cs_s['duration_sec']:.2f}")
# lost chunks: same jump, header reports skipped chunks -> a gap
h2 = dict(h, skipped_chunks=3)
h2["clock_summary"] = dict(h["clock_summary"], clock_steps=[], n_clock_steps=0)
stp.write_bytes(_hdr_bytes(h2) + rec.tobytes())
cg = pc.read_ppd(stp)["clock"]
check("jump with dropped chunks: reported as lost samples", len(cg["gaps"]) == 1 and not cg["clock_steps"]
      and any("dropped 3" in w for w in cg["warnings"]))

# old-style (performance-counter) stamps keep the sample clock
old = tmp / "old.ppd"
write_ppd(old, 1794931686.0, 4000, 20.0, np.full(4000, 1.2), np.full(4000, 0.5),
          extra={"pc_clock": {"format": "float64", "anchor_unix_s": 1794931686.0, "utc_offset_s": UTC_OFF}})
check("files without pc_clock.source keep the sample clock", pc.read_ppd(old)["clock"]["source"] == "sample")

# a NUL-padded header (interrupted rewrite) still opens
nul = tmp / "nul.ppd"
b = old.read_bytes()
hl = int.from_bytes(b[:2], "little")
body = b[2:2 + hl].rstrip(b" ")
nul.write_bytes(b[:2] + body + b"\x00" * (hl - len(body)) + b[2 + hl:])
check("NUL-padded header opens", pc.read_ppd(nul)["n_samples"] == 4000)

print("\n=== pyPhotometry CSV recording (.csv + .json) ===")
cdir = tmp / "csvrec"
cdir.mkdir()
k = np.arange(2400)
stamps = 1794931686.0 + k / 20.0 + 0.003
hdr = {"subject_ID": "m1", "date_time": "2026-11-17T09:28:06.000", "end_time": "2026-11-17T09:30:06.400",
       "mode": "2EX_1EM_pulsed", "sampling_rate": 20, "volts_per_division": [VPD, VPD], "version": "1.1",
       "pc_timestamps": True, "pc_clock": {"source": "system_clock", "utc_offset_s": UTC_OFF}}
(cdir / "m1-2026-11-17-092806.json").write_text(json.dumps(hdr, indent=4))
a1 = np.round(1.3 / VPD + 50 * np.sin(k / 30)).astype(int)
a2 = np.round(0.5 / VPD + np.zeros(k.size)).astype(int)
lines = ["Analog1, Analog2, Digital1, Digital2, PC_timestamp"] + [
    f"{x}, {y}, 0, {int(i % 200 == 0)}, {t:.6f}" for i, (x, y, t) in enumerate(zip(a1, a2, stamps))]
(cdir / "m1-2026-11-17-092806.csv").write_text("\n".join(lines) + "\n")
rc = intake.classify(intake.expand([cdir], tmp / "x"))
check("CSV recording classified as photometry, its .json not listed as skipped",
      [p.name for p in rc["photometry"]] == ["m1-2026-11-17-092806.csv"] and not rc["skipped"], str(rc))
dc = pc.read_pyphotometry_csv(cdir / "m1-2026-11-17-092806.csv")
check("CSV recording read: counts -> volts, 20 Hz, system clock",
      abs(np.median(dc["analog_1"]) - 1.3) < 0.01 and dc["n_samples"] == 2400 and dc["clock"]["source"] == "pc",
      f"{np.median(dc['analog_1']):.3f} V")
check("digital pulses kept", int(dc["digital_2"].sum()) == 12)

print("\n=== baseline normalisation ===")
t = np.arange(0, 480, 0.05)
rng = np.random.default_rng(3)
y = 0.5 * rng.standard_normal(t.size)                       # dF/F %, 5 min flat
pk = (t >= 300) & (t < 480)
y = y + np.where(pk, 6.0 * np.exp(-((t - 330) / 40.0) ** 2), 0)    # peak, back in ~3 min
r = bl.peak_vs_baseline(t, y, 300.0, 300.0, 180.0, smooth_sec=1.0)
check("peak found ~30 s after the baseline", r["ok"] and 25 < r["peak_time_s"] < 35, f"{r.get('peak_time_s')}")
check("baseline-only z is large (peak 6 % over SD 0.5 %)", 9 < r["peak_z"] < 14, f"{r['peak_z']:.2f}")
check("whole-window z of the same peak is much smaller", r["peak_z_whole_window"] < 0.6 * r["peak_z"],
      f"{r['peak_z_whole_window']:.2f}")
check("returns to baseline within the window", r["return_s"] is not None and 60 < r["return_s"] < 150,
      f"{r['return_s']}")
check("% change from baseline uses the corrected fluorescence",
      abs(r["peak_change_pct"] - ((1 + r["peak_dff_pct"] / 100) / (1 + r["baseline_mean_pct"] / 100) - 1) * 100) < 1e-9)
s_ = bl.baseline_stats(t, y, 300.0, 300.0)
z = bl.z_to_baseline(y, s_)
check("z to baseline: mean 0 and SD 1 inside the baseline",
      abs(np.mean(z[t < 300])) < 1e-9 and abs(np.std(z[t < 300], ddof=1) - 1) < 1e-9)
check("baseline outside the recording -> not ok", not bl.baseline_stats(t, y, -10.0, 60.0)["ok"])

import group_analysis as ga
m = ga.make_member("r", "a", "All", t, y + 5.0, pd.Timestamp("2026-11-17 09:00"))
bm = ga.baseline_mask(m, "Time since recording start", 5.0, end_x=300.0)
check("group baseline ending at a chosen time", abs(t[bm].min() - 0) < 0.06 and t[bm].max() < 300)
pc_ = ga.normalise(t, y + 5.0, "% change from baseline mean", bm, dff_percent=True)
check("group % change of dF/F is (1+y)/(1+b)-1", abs(np.nanmean(pc_[t < 300])) < 1e-6)

print("\n=== longitudinal: actogram and meal size ===")
tt = np.arange(0, 2.5 * 86400, 60.0)
cs = pd.Timestamp("2026-11-17 09:00")
yy = np.sin(2 * np.pi * ((tt / 3600 + 9) % 24) / 24)
A = lg.actogram(tt, yy, cs, 600.0, pellets=np.array([3600.0, 90000.0]))
check("actogram: 3 calendar days, 48 h wide", len(A["day_labels"]) == 3 and A["M"].shape == (3, 288),
      str(A["M"].shape))
check("double plot: row d right half == row d+1 left half",
      np.allclose(A["M"][0, 144:], A["M"][1, :144], equal_nan=True))
check("each pellet drawn twice (row and row above)", len(A["ticks_x"]) == 3, str(A["ticks_x"]))
check("dark 10 AM-10 PM on the 48 h axis", lg.dark_spans_48h(22.0, 10.0) == [(10.0, 22.0), (34.0, 46.0)])
mm = ga.make_member("r", "a", "All", t, np.zeros(t.size), cs)
tm = np.arange(0, 7200, 1.0)
yy = np.zeros(tm.size)
pel = []
for on, k_ in ((1000, 1), (2500, 3), (4000, 5), (5500, 7)):
    pel += [on + 20 * j for j in range(k_)]
    yy -= 0.3 * k_ * (tm >= on)
mm = ga.make_member("r", "a", "All", tm, yy, cs)
mm["pellets"] = np.array(pel, float)
df = lg.meal_responses(mm, 60.0, 1, (-300, -30), (0, 300))
check("one row per meal with its size", list(df["pellets"]) == [1, 3, 5, 7], str(list(df["pellets"])))
check("response = drop proportional to size", np.allclose(df["response"], [-0.3, -0.9, -1.5, -2.1]))
per, grp = lg.slope_stats(df)
check("slope per pellet -0.3, rho -1", abs(per["slope per pellet"].iat[0] + 0.3) < 1e-9 and per["rho"].iat[0] < -0.99)

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
