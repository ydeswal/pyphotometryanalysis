"""Validation tests for photometry_core."""
import json, struct, sys
import numpy as np
import photometry_core as pc

VPD = 3.3 / 32768
rng = np.random.default_rng(0)


def write_ppd(path, mode, version, n_samples, ch1_on, ch1_off, ch2_on, ch2_off,
              dig1=None, dig2=None, pulsed_new=True, fs=130.0):
    header = {
        "subject_ID": "test", "date_time": "2026-01-01T00:00:00", "mode": mode,
        "sampling_rate": fs, "version": version,
        "volts_per_division": [VPD, VPD],
        "n_analog_channels": 2, "n_digital_channels": 2,
    }
    hb = json.dumps(header).encode("utf-8")
    to_u16 = lambda v, d: ((np.round(v / VPD).astype(np.int64) << 1) | d).astype("<u2")
    if pulsed_new:
        w = np.empty(n_samples * 4, dtype="<u2")
        w[0::4] = to_u16(ch1_on, dig1 if dig1 is not None else 0)
        w[1::4] = to_u16(ch1_off, 0)
        w[2::4] = to_u16(ch2_on, dig2 if dig2 is not None else 0)
        w[3::4] = to_u16(ch2_off, 0)
    else:
        w = np.empty(n_samples * 2, dtype="<u2")
        w[0::2] = to_u16(ch1_on, dig1 if dig1 is not None else 0)
        w[1::2] = to_u16(ch2_on, dig2 if dig2 is not None else 0)
    with open(path, "wb") as f:
        f.write(struct.pack("<H", len(hb))); f.write(hb); f.write(w.tobytes())
    return header


def approx(a, b, tol=1e-6):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) < tol


fails = []
def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


N = 4000
t = np.arange(N) / 130.0
# Distinct, recoverable ground truth per stream.
g_on   = 0.90 + 0.05 * np.sin(2 * np.pi * 0.05 * t)
g_off  = 0.10 + 0.01 * np.cos(2 * np.pi * 0.03 * t)
i_on   = 0.50 + 0.02 * np.sin(2 * np.pi * 0.07 * t)
i_off  = 0.05 + 0.005 * np.cos(2 * np.pi * 0.02 * t)
lick   = (rng.random(N) < 0.02).astype(np.uint8)

print("\n=== 1. NEW-FORMAT PULSED .ppd (v1.1+) ===")
write_ppd("t_new.ppd", "2EX_1EM_pulsed", "1.1.1", N, g_on, g_off, i_on, i_off, dig1=lick)
d = pc.read_ppd("t_new.ppd", keep_components=True, with_time=True)
check("frame layout detected", d["frame_layout"] == "pulsed_v1.1_led_on_off", d["frame_layout"])
check("analog_1 == 465on - 465off", approx(d["analog_1"], g_on - g_off, 1e-4))
check("analog_2 == 405on - 405off", approx(d["analog_2"], i_on - i_off, 1e-4))
check("LED-on component preserved", approx(d["analog_1_raw_LED_on"], g_on, 1e-4))
check("digital_1 recovered (lickometer TTL)", np.array_equal(d["digital_1"], lick))
check("sample count", len(d["analog_1"]) == N, f"{len(d['analog_1'])}")
d_lean = pc.read_ppd("t_new.ppd")
check("LED components omitted by default (saves memory)",
      "analog_1_raw_LED_on" not in d_lean)
check("lean read still correct", approx(d_lean["analog_1"], g_on - g_off, 1e-4))
check("signals are float32", d_lean["analog_1"].dtype == np.float32,
      str(d_lean["analog_1"].dtype))
check("time axis omitted by default", "time_sec" not in d_lean)
check("time axis float64 when requested", d["time_sec"].dtype == np.float64)

print("\n=== 2. OLD-FORMAT / CONTINUOUS .ppd (2 words per frame) ===")
write_ppd("t_old.ppd", "2EX_2EM_continuous", "1.0.0", N, g_on, None, i_on, None,
          dig1=lick, pulsed_new=False)
d2 = pc.read_ppd("t_old.ppd")
check("frame layout detected", d2["frame_layout"] == "one_word_per_channel", d2["frame_layout"])
check("analog_1 == channel 1", approx(d2["analog_1"], g_on, 1e-4))
check("analog_2 == channel 2", approx(d2["analog_2"], i_on, 1e-4))
check("digital_1 recovered", np.array_equal(d2["digital_1"], lick))

print("\n=== 3. WHAT THE OLD ana.py CODE DID TO THE SAME OLD-FORMAT FILE ===")
# Reproduce the previous reader exactly: always reshape to n_analog+n_digital = 4.
with open("t_old.ppd", "rb") as f:
    hs = int.from_bytes(f.read(2), "little"); json.loads(f.read(hs)); rawb = f.read()
raw = np.frombuffer(rawb, dtype="<u2")
old = (raw.reshape(-1, 4) >> 1).astype(float) * VPD
old_sig = old[:, 0] - old[:, 1]   # "465_on - 465_dark"
old_uv  = old[:, 2] - old[:, 3]   # "405_on - 405_dark"
corr = np.corrcoef(old_sig, old_uv)[0, 1]
print(f"    old reader 'signal' vs 'isosbestic' correlation: r = {corr:.4f}")
print(f"    true  465 vs 405 correlation:                    r = "
      f"{np.corrcoef(g_on, i_on)[0,1]:.4f}")
check("old reader corrupts old-format files (r>0.99 = same quantity twice)",
      corr > 0.99, f"r={corr:.4f}")
check("old reader also halves the sample count", len(old_sig) == N // 2,
      f"got {len(old_sig)} samples, true N={N} -> recording duration halved")

print("\n=== 4. volts_per_division indexing ===")
# Old code used vpd[min(i, len(vpd)-1)] -> cols got vpd[0],vpd[1],vpd[1],vpd[1].
# Column 1 (the 465 dark baseline) got channel 2's scale factor.
idx_old = [min(i, 1) for i in range(4)]
check("old vpd mapping was wrong for column 1", idx_old == [0, 1, 1, 1],
      f"cols->vpd {idx_old}, correct is [0,0,1,1]")

print("\n=== 5. crop_and_rezero ===")
import pandas as pd
df = pd.DataFrame({"time_sec": np.arange(0, 172800, 1.0)})
df["elapsed_hours"] = df.time_sec / 3600
cropped, off = pc.crop_and_rezero(df, start_sec=86400, end_sec=None)
check("offset == requested start", off == 86400.0, f"{off}")
check("hour 24 becomes t=0", cropped.time_sec.iloc[0] == 0.0, f"{cropped.time_sec.iloc[0]}")
check("hour 25 becomes hour 1", abs(cropped.elapsed_hours.iloc[3600] - 1.0) < 1e-9)
check("last sample = 24 h", abs(cropped.elapsed_hours.iloc[-1] - 23.99972) < 1e-3,
      f"{cropped.elapsed_hours.iloc[-1]:.5f}")
check("original timeline retained", cropped.original_time_sec.iloc[0] == 86400.0)

print("\n=== 6. very-low-cutoff F0 filter stability (0.001 Hz) ===")
fs = 130.0
n = int(fs * 3600 * 6)            # 6 hours
tt = np.arange(n) / fs
slow = 1.0 * np.exp(-tt / 20000.0)          # photobleaching
fast = 0.05 * np.sin(2 * np.pi * 0.5 * tt)  # 0.5 Hz calcium-like
sig = slow + fast
import time as _time
t0 = _time.time(); f0 = pc.compute_f0(sig, fs, method="lowpass", lowpass_hz=0.001)
el = _time.time() - t0
check("F0 all finite", np.all(np.isfinite(f0)))
check("F0 tracks slow envelope, rejects 0.5 Hz",
      np.std(f0 - slow) < 0.02, f"resid sd={np.std(f0-slow):.5f}")
check("F0 computed fast", el < 5.0, f"{el:.2f}s for 6 h @130Hz")

print("\n=== 7. median filter speed and edges ===")
x = rng.normal(size=int(130 * 3600 * 2))    # 2 hours
t0 = _time.time(); m = pc.median_filter_trace(x, 130.0, 1.0); el = _time.time() - t0
check("median filter fast", el < 10.0, f"{el:.2f}s for 2 h @130Hz")
check("no zero-padding at edges", abs(m[0]) < 1.0 and abs(m[-1]) < 1.0,
      f"first={m[0]:.4f} last={m[-1]:.4f}")

print("\n=== 8. IRLS vs OLS with real transients present ===")
n = 20000
tt = np.arange(n) / 130.0
motion = 0.3 * np.sin(2 * np.pi * 0.02 * tt) + 0.02 * rng.normal(size=n)
control = 1.0 + motion
true_slope = 0.8
transients = np.zeros(n)
for s in range(500, n - 500, 1500):          # real calcium events, control-independent
    transients[s:s + 300] += 0.25 * np.exp(-np.arange(300) / 80.0)
signal = 2.0 + true_slope * motion + transients
_, s_ols, _, _ = pc.fit_control_to_signal(control, signal, method="ols")
_, s_irls, _, _ = pc.fit_control_to_signal(control, signal, method="irls")
print(f"    true slope {true_slope:.4f} | OLS {s_ols:.4f} | IRLS {s_irls:.4f}")
check("IRLS closer to true slope than OLS",
      abs(s_irls - true_slope) < abs(s_ols - true_slope),
      f"err OLS={abs(s_ols-true_slope):.4f} IRLS={abs(s_irls-true_slope):.4f}")

print("\n=== 9. z-score modes ===")
v = np.concatenate([rng.normal(0, 1, 1000), rng.normal(5, 1, 200)])
bmask = np.zeros(len(v), bool); bmask[:1000] = True
zs = pc.zscore(v); zb = pc.zscore(v, baseline_mask=bmask); zr = pc.zscore(v, robust=True)
check("session z has mean~0 sd~1", abs(np.mean(zs)) < 1e-9 and abs(np.std(zs, ddof=1) - 1) < 1e-9)
check("baseline z: baseline epoch sd~1", abs(np.std(zb[bmask], ddof=1) - 1) < 1e-9)
check("baseline z: event epoch elevated ~5", 4 < np.mean(zb[~bmask]) < 6,
      f"{np.mean(zb[~bmask]):.2f}")
check("baseline z > session z for events", np.mean(zb[~bmask]) > np.mean(zs[~bmask]))
check("robust z finite", np.all(np.isfinite(zr)))

print("\n=== 10. full pipeline ===")
n = 30000; fs = 130.0; tt = np.arange(n)/fs
ctl = 1.0 + 0.2*np.sin(2*np.pi*0.01*tt) + 0.01*rng.normal(size=n)
sg = 2.0 + 0.16*np.sin(2*np.pi*0.01*tt) + 0.01*rng.normal(size=n)
sg[5000:5500] += 0.3
res = pc.process_photometry(sg, ctl, fs, {"f0_method": "percentile", "median_filter_sec": 0})
check("dFF finite", np.mean(np.isfinite(res["dFF"])) > 0.99)
check("z finite", np.mean(np.isfinite(res["z_dFF"])) > 0.99)
check("transient detected in z", np.nanmax(res["z_dFF"][5000:5500]) > 2.0,
      f"peak z={np.nanmax(res['z_dFF'][5000:5500]):.2f}")
check("F0 from signal not control", abs(res["F0"] - np.nanpercentile(sg, 10)) < 0.15,
      f"F0={res['F0']:.3f} vs signal p10={np.nanpercentile(sg,10):.3f}")

print("\n=== 11. OLS matches scipy.stats.linregress (the paper's 'linear best fit') ===")
from scipy.stats import linregress
lr = linregress(control, signal)
_, s_o, i_o, info = pc.fit_control_to_signal(control, signal, method="ols")
check("slope identical to linregress", abs(s_o - lr.slope) < 1e-9, f"{s_o:.6f} vs {lr.slope:.6f}")
check("intercept identical", abs(i_o - lr.intercept) < 1e-9)
check("R^2 identical", abs(info["r2"] - lr.rvalue**2) < 1e-9, f"{info['r2']:.4f}")

print("\n=== 12. negative-slope guard ===")
n = 4000; tt = np.arange(n) / 20.0
real = np.where(tt > 90, 0.8, 0.0)                    # big real 465 increase
ctl_n = 0.73 - 0.03 * real / 0.8 + 0.003 * rng.normal(size=n)   # 405 dips slightly
sig_n = 1.46 + real + 0.005 * rng.normal(size=n)
_, sl, ic, inf = pc.fit_control_to_signal(ctl_n, sig_n, negative_slope_policy="allow")
check("unconstrained fit is negative and large", sl < -5, f"slope={sl:.2f}")
fit_z, sl0, ic0, inf0 = pc.fit_control_to_signal(ctl_n, sig_n, negative_slope_policy="zero")
check("clamped slope = 0", sl0 == 0.0 and inf0["slope_clamped"])
check("clamped intercept = mean(465)", abs(ic0 - sig_n.mean()) < 1e-12)
dF_allow = sig_n - (sl * ctl_n + ic); dF_zero = sig_n - fit_z
step_allow = dF_allow[tt > 100].mean() - dF_allow[tt < 80].mean()
step_zero = dF_zero[tt > 100].mean() - dF_zero[tt < 80].mean()
check("negative slope erases most of the real step", abs(step_allow) < 0.2, f"{step_allow:.3f}")
check("clamped fit preserves the real step", abs(step_zero - 0.8) < 0.02, f"{step_zero:.3f}")

print("\n=== 13. F0 edge behaviour on a short recording ===")
f0s = pc.compute_f0(sig_n, 20.0)
check("F0 not pinned to first sample", abs(f0s[0] - sig_n[:20].mean()) > 0.2, f"F0[0]={f0s[0]:.3f}")
check("F0 ~ session mean when record < 1/cutoff", np.max(np.abs(f0s - sig_n.mean())) < 0.05,
      f"max dev={np.max(np.abs(f0s - sig_n.mean())):.4f}")

print("\n=== 14. low-pass at Nyquist is the identity ===")
x = rng.normal(size=1000)
check("10 Hz low-pass at 20 Hz returns input", np.array_equal(pc.lowpass(x, 20.0, 10.0), x))

print("\n=== 15. blocked filtering equals one-shot filtering ===")
old_block = pc._FILTER_BLOCK
y = np.cumsum(rng.normal(size=60000)).astype(np.float64)
one = pc.lowpass(y, 130.0, 2.0)
pc._FILTER_BLOCK = 15000
blk = pc.lowpass(y, 130.0, 2.0)
pc._FILTER_BLOCK = old_block
check("block result matches", np.max(np.abs(one - blk)) < 1e-6 * np.ptp(y),
      f"max diff={np.max(np.abs(one - blk)):.2e}")

print("\n=== 16. dF/F definition ===")
r16 = pc.process_photometry(sg, ctl, fs, {"median_kernel_samples": 0})
check("dFF == deltaF / F0_trace", np.allclose(r16["dFF"], r16["deltaF"] / r16["F0_trace"], equal_nan=True))
check("deltaF == sig_filt - ctl_fit", np.allclose(r16["deltaF"], r16["sig_filt"] - r16["ctl_fit"]))
zz = r16["z_dFF"]
check("z = (dFF-mean)/SD", np.allclose(zz, (r16["dFF"] - np.nanmean(r16["dFF"])) / np.nanstd(r16["dFF"], ddof=1)))

print("\n=== 17. pyPhotometry pipeline recovers bleaching + motion ===")
n = int(40 * 60 * 20); fs17 = 20.0; t17 = np.arange(n) / fs17
bleach_s = 1.0 + 0.4 * np.exp(-t17 / 900)
bleach_c = 0.6 + 0.1 * np.exp(-t17 / 1200)
mot = 0.02 * np.sin(2 * np.pi * 0.05 * t17)
ev = np.zeros(n); ev[(t17 % 127) < 5] = 0.05
sig17 = bleach_s * (1 + ev) + 0.5 * mot + 0.002 * rng.normal(size=n)
ctl17 = bleach_c + mot + 0.002 * rng.normal(size=n)
r17 = pc.process_photometry(sig17, ctl17, fs17, {"pipeline": "pyphotometry"})
check("pyphotometry pipeline ran", r17["pipeline_used"] == "pyphotometry")
check("motion slope recovered (~0.5)", abs(r17["slope"] - 0.5) < 0.05, f"{r17['slope']:.3f}")
on = r17["dFF"][(t17 % 127) < 5].mean() - r17["dFF"][(t17 % 127) > 20].mean()
check("event dF/F recovered (~5%)", abs(on - 0.05) < 0.01, f"{100*on:.2f}%")

print("\n=== 18. the uploaded M16 recording (if present) ===")
import os
P = "/mnt/user-data/uploads/M16_KEREM-2026-10-02-114706.ppd"
if os.path.exists(P):
    d = pc.read_ppd(P)
    r18 = pc.process_photometry(d["analog_1"], d["analog_2"], d["sampling_rate"])
    check("negative slope detected and clamped", r18["fit_info"]["slope_clamped"],
          f"raw slope={r18['fit_info']['raw_slope']:.2f}")
    check("F0 ~ session mean (1.79 V)", abs(r18["F0"] - float(np.mean(d["analog_1"]))) < 0.03,
          f"F0={r18['F0']:.3f}")
else:
    print("  (skipped)")

print("\n=== 19. PC-timestamped .ppd (16 bytes per sample) ===")


def write_ppd_pc(path, n, fs=20.0, declare=True, drift_ppm=27.6, jitter=0.003,
                 gap_at=None, gap_sec=0.0, step_back_at=None, tail_bytes=0,
                 start_unix=1_791_991_380.4, dig1=None, end_time=None, utc_off=-18000):
    """Pulsed v1.1 frames, each followed by a float64 PC 'arrival' timestamp."""
    tt = np.arange(n) / fs
    on1 = 0.9 + 0.05 * np.sin(2 * np.pi * 0.01 * tt); off1 = 0.02 + 0 * tt
    on2 = 0.6 + 0.04 * np.sin(2 * np.pi * 0.01 * tt); off2 = 0.01 + 0 * tt
    to_u16 = lambda v, d: ((np.round(v / VPD).astype(np.int64) << 1) | d).astype("<u2")
    words = np.empty((n, 4), "<u2")
    words[:, 0] = to_u16(on1, dig1 if dig1 is not None else 0)
    words[:, 1] = to_u16(off1, 0); words[:, 2] = to_u16(on2, 0); words[:, 3] = to_u16(off2, 0)
    pc_t = start_unix + tt * (1 + drift_ppm * 1e-6) + np.abs(rng.normal(0, jitter, n))
    if gap_at is not None:
        pc_t[gap_at:] += gap_sec
    if step_back_at is not None:
        pc_t[step_back_at:] -= 1.5
    rec = np.empty(n, dtype=[("w", "<u2", (4,)), ("t", "<f8")])
    rec["w"] = words; rec["t"] = pc_t
    hdr = {"subject_ID": "pc", "date_time": "2026-10-14T10:23:00.385", "mode": "2EX_1EM_pulsed",
           "sampling_rate": fs, "version": "1.1", "volts_per_division": [VPD, VPD],
           "n_analog_signals": 2, "n_digital_signals": 2, "ADC_max_value": 32768}
    if end_time:
        hdr["end_time"] = end_time
    if declare:
        hdr.update({"pc_timestamps": True, "words_per_sample": 4, "bytes_per_sample": 16,
                    "pc_clock": {"format": "unix_epoch_seconds_float64",
                                 "anchor_unix_s": start_unix, "utc_offset_s": utc_off}})
    hb = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<H", len(hb))); f.write(hb); f.write(rec.tobytes())
        f.write(b"\x00" * tail_bytes)
    return on1 - off1, on2 - off2, pc_t


n19 = 20 * 3600 * 2          # 2 h at 20 Hz
lick19 = (rng.random(n19) < 0.01).astype(np.uint8)
s1, s2, pct = write_ppd_pc("t_pc.ppd", n19, dig1=lick19)
d = pc.read_ppd("t_pc.ppd", with_time=True)
c = d["clock"]
check("sample count not doubled", d["n_samples"] == n19, f"{d['n_samples']} vs {n19}")
check("duration correct (2 h, not 4 h)", abs(d["duration_sec"] - 7200) < 0.1, f"{d['duration_sec']:.1f}")
check("analog_1 recovered", approx(d["analog_1"], s1, 1e-4))
check("analog_2 recovered", approx(d["analog_2"], s2, 1e-4))
check("digital recovered", np.array_equal(d["digital_1"], lick19))
check("no fake clipping", d["analog_1_clipping_fraction"] == 0.0)
check("start time = first acquisition, within 5 ms", abs(c["start_unix"] - 1_791_991_380.4) < 0.005,
      f"{(c['start_unix'] - 1_791_991_380.4) * 1e3:.2f} ms")
check("drift measured (27.6 ppm)", abs(c["pc_drift_ppm"] - 27.6) < 1.0, f"{c['pc_drift_ppm']:.2f}")
check("default time axis is the sample clock", np.allclose(d["time_sec"], np.arange(n19) / 20.0))
check("no gaps reported", c["gaps"] == [])
cols = pc.clock_columns(c, d["time_sec"][[0, -1]])
check("clock_time in local time (UTC-5)", cols["clock_time"][0].startswith("2026-10-14 10:23:00"),
      cols["clock_time"][0])
dp = pc.read_ppd("t_pc.ppd", with_time=True, clock_source="pc")
check("pc time axis follows the stamps (jitter removed)",
      np.max(np.abs(dp["time_sec"] + dp["clock"]["start_unix"] - pct)) < 0.03,
      f"max dev {np.max(np.abs(dp['time_sec'] + dp['clock']['start_unix'] - pct)) * 1e3:.1f} ms")

write_ppd_pc("t_pc_undeclared.ppd", n19, declare=False)
du = pc.read_ppd("t_pc_undeclared.ppd")
check("undeclared timestamps detected from the data", du["clock"]["has_pc_timestamps"]
      and du["n_samples"] == n19, du.get("layout_note", ""))

write_ppd_pc("t_pc_trunc.ppd", 4000, tail_bytes=11)
dt_ = pc.read_ppd("t_pc_trunc.ppd")
check("interrupted file: partial last sample dropped, rest recovered",
      dt_["n_samples"] == 4000 and any("incomplete" in w for w in dt_["clock"]["warnings"]))

s1g, _, _ = write_ppd_pc("t_pc_gap.ppd", n19, gap_at=50_000, gap_sec=12.0)
dg = pc.read_ppd("t_pc_gap.ppd", with_time=True)
g = dg["clock"]["gaps"]
check("lost samples detected", len(g) == 1 and abs(g[0]["gap_sec"] - 12.0) < 0.05,
      str(g))
check("gap located to the sample", len(g) == 1 and abs(g[0]["at_sample"] - 50_000) <= 1,
      str(g[0]["at_sample"]) if g else "")
check("time axis jumps over the gap",
      abs(dg["time_sec"][50_000] - dg["time_sec"][49_999] - (0.05 + 12.0)) < 0.06,
      f"{dg['time_sec'][50_000] - dg['time_sec'][49_999]:.3f} s")
ev = pc.digital_pulse_times(np.r_[np.zeros(60_000, np.uint8), 1, np.zeros(9, np.uint8)], 20.0,
                            clock=dg["clock"])
check("digital event after a gap gets the gap-corrected time",
      abs(ev[0] - (60_000 / 20.0 + 12.0)) < 0.06, f"{ev}")

write_ppd_pc("t_pc_back.ppd", n19, step_back_at=60_000)
db = pc.read_ppd("t_pc_back.ppd", with_time=True)
check("PC clock set back: not treated as lost samples", db["clock"]["gaps"] == []
      and np.allclose(db["time_sec"], np.arange(n19) / 20.0))

# 10:23:00.4 local + 2 h = 12:23:00.35 last sample on the sample clock; stamps
# drifting 300 ppm put it 2.2 s later, i.e. after the wall-clock end_time.
write_ppd_pc("t_pc_end.ppd", n19, end_time="2026-10-14T12:23:00.700",
             start_unix=1_791_991_380.4, drift_ppm=300)
de = pc.read_ppd("t_pc_end.ppd")
chk = de["clock"]["end_time_check"]
check("end_time cross-check agrees with the sample clock",
      chk is not None and 0 < chk["end_minus_last_sample_sec"] < 1, str(chk))
dep = pc.read_ppd("t_pc_end.ppd", clock_source="pc")
check("end_time cross-check flags drifted PC stamps",
      any("end_time" in w for w in dep["clock"]["warnings"]))

print("\n=== 20. standard files unaffected; truncated standard file not misread ===")
write_ppd("t_new_trunc.ppd", "2EX_1EM_pulsed", "1.1.1", N, g_on, g_off, i_on, i_off, dig1=lick)
with open("t_new_trunc.ppd", "ab") as f:
    f.write(b"\x10\x00\x20\x00")            # half a frame: 2 words of 4
d20 = pc.read_ppd("t_new_trunc.ppd")
check("layout kept (no fall-back to 2 words/frame)",
      d20["frame_layout"] == "pulsed_v1.1_led_on_off" and d20["n_samples"] == N,
      f"{d20['frame_layout']} n={d20['n_samples']}")
check("data intact", approx(d20["analog_1"], g_on - g_off, 1e-4))
d20b = pc.read_ppd("t_new.ppd")
check("standard file: no PC timestamps, start from header",
      not d20b["clock"]["has_pc_timestamps"] and d20b["clock"]["start_local"].startswith("2026-01-01"))

print("\n=== 21. convert_ppd round trip ===")
import convert_ppd, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    outp = convert_ppd.convert("t_pc_gap.ppd", outdir="t_conv")
dc = pc.read_ppd(outp, with_time=True)
check("converted file is standard size (8 bytes/sample)",
      os.path.getsize(outp) < os.path.getsize("t_pc_gap.ppd") * 0.55)
check("converted data identical", approx(dc["analog_1"], s1g, 1e-4))
check("converted file keeps start time", abs(dc["clock"]["start_unix"] - dg["clock"]["start_unix"]) < 1e-6)
check("converted file keeps the gap", np.allclose(dc["time_sec"], dg["time_sec"]))
ts = pd.read_csv("t_conv/t_pc_gap_pc_timestamps.csv.gz")
check("timestamps sidecar complete", len(ts) == n19 and list(ts.columns) == ["sample", "pc_unix"])

print("\n=== 22. app: lick-file clock alignment helper ===")
import ast
src = open("app.py").read()
fn = next(nd for nd in ast.parse(src).body
          if isinstance(nd, ast.FunctionDef) and nd.name == "_clock_offset_sec")
ns = {"pd": pd}
exec(compile(ast.Module([fn], []), "app.py", "exec"), ns)
off = ns["_clock_offset_sec"](pd.Timestamp("2026-10-14 11:00:00"),
                              {"start_clock_local": "2026-10-14T10:23:00.423", "utc_offset_s": -18000})
check("naive local clocks", abs(off - (36 * 60 + 59.577)) < 1e-6, f"{off}")
off = ns["_clock_offset_sec"](pd.Timestamp("2026-10-14 16:00:00", tz="UTC"),
                              {"start_clock_local": "2026-10-14T10:23:00.423", "utc_offset_s": -18000})
check("UTC lick clock converted to the recording's offset", abs(off - (36 * 60 + 59.577)) < 1e-6,
      f"{off}")
check("no start clock -> no alignment",
      ns["_clock_offset_sec"](pd.Timestamp("2026-10-14 11:00"), {}) is None)

print("\n=== 23. a real PC-timestamped recording (if present) ===")
import glob
cands = glob.glob("m10_fasted*.ppd") + glob.glob("/mnt/user-data/uploads/*m10_fasted*.ppd")
if cands:
    d = pc.read_ppd(cands[0])
    r23 = pc.process_photometry(d["analog_1"], d["analog_2"], d["sampling_rate"])
    check("23.93 h, not 47.86 h", abs(d["duration_sec"] / 3600 - 23.93) < 0.01,
          f"{d['duration_sec'] / 3600:.3f} h")
    check("no clipping (the 2.09% was timestamp bytes)", d["analog_1_clipping_fraction"] == 0.0)
    check("positive 405 slope, good fit", r23["slope"] > 0 and r23["fit_info"]["r2"] > 0.8,
          f"slope={r23['slope']:.3f} R2={r23['fit_info']['r2']:.3f}")
    check("no lost samples", d["clock"]["gaps"] == [])
    check("end_time 0-1 s after last sample",
          0 < d["clock"]["end_time_check"]["end_minus_last_sample_sec"] < 1)
else:
    print("  (skipped)")

print("\n" + "="*60)
print(f"RESULT: {'ALL TESTS PASSED' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
