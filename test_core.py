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
d = pc.read_ppd("t_new.ppd")
check("frame layout detected", d["frame_layout"] == "pulsed_v1.1_led_on_off", d["frame_layout"])
check("analog_1 == 465on - 465off", approx(d["analog_1"], g_on - g_off, 1e-4))
check("analog_2 == 405on - 405off", approx(d["analog_2"], i_on - i_off, 1e-4))
check("LED-on component preserved", approx(d["analog_1_raw_LED_on"], g_on, 1e-4))
check("digital_1 recovered (lickometer TTL)", np.array_equal(d["digital_1"], lick))
check("sample count", len(d["analog_1"]) == N, f"{len(d['analog_1'])}")

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
_, s_ols, _ = pc.fit_control_to_signal(control, signal, method="ols")
_, s_irls, _ = pc.fit_control_to_signal(control, signal, method="irls")
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

print("\n" + "="*60)
print(f"RESULT: {'ALL TESTS PASSED' if not fails else str(len(fails)) + ' FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
