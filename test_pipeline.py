"""Ground-truth tests of the dF/F pipelines on simulated recordings (no data files needed).

Each simulation builds   signal = bg + F_s * bleach_s(t) * motion(t) * (1 + true_dff(t))
                         control = bg + F_c * bleach_c(t) * motion(t) * (1 + leak * true_dff(t))
so the true dF/F is known, and checks how closely each pipeline recovers it."""
import numpy as np

import photometry_core as pc

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


def simulate(T=600.0, fs=20.0, seed=0, leak=0.0, bg=0.0, step=None, motion_amp=0.05, bleach=(0.10, 0.08),
             transients=True, noise=0.002):
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1 / fs)
    true = np.zeros(t.size)
    if transients:
        for t0 in rng.uniform(30, T - 30, int(T / 40)):
            true += rng.uniform(0.05, 0.15) * (t >= t0) * np.exp(-np.clip(t - t0, 0, None) / rng.uniform(3, 10))
    if step:
        t0, amp = step
        true += amp * (t >= t0) * (1 - np.exp(-np.clip(t - t0, 0, None) / 5))
    slow = np.convolve(rng.standard_normal(t.size), np.ones(int(fs * 2)) / (fs * 2), mode="same")
    motion = 1 + motion_amp * slow / (np.std(slow) + 1e-12) * 0.3
    for t0 in rng.uniform(20, T - 20, int(T / 60)):          # brief movement dips
        motion *= 1 - motion_amp * 2 * np.exp(-((t - t0) / 1.5) ** 2)
    bs = 1 - bleach[0] * (1 - np.exp(-t / (T / 3)))
    bc = 1 - bleach[1] * (1 - np.exp(-t / (T / 3)))
    sig = bg + 1.5 * bs * motion * (1 + true) * (1 + noise * rng.standard_normal(t.size))
    ctl = bg + 0.8 * bc * motion * (1 + leak * true) * (1 + noise * rng.standard_normal(t.size))
    return t, sig, ctl, true, fs


def err(est, true, t):
    """RMS error of the recovered dF/F against the truth, both centred on their median
    (the zero level of dF/F is arbitrary), ignoring the first 10 s."""
    k = t > 10
    a = est[k] - np.nanmedian(est[k])
    b = true[k] - np.median(true[k])
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def run(sim, pipeline, **kw):
    t, sig, ctl, true, fs = sim
    r = pc.process_photometry(sig, ctl, fs, settings=dict(pipeline=pipeline, **kw))
    return r, err(r["dFF"], true, t)


print("=== valid control: transients + movement + bleaching ===")
e_old, e_new = [], []
for seed in range(6):
    sim = simulate(seed=seed)
    e_old.append(run(sim, "douglass2025")[1])
    r, e = run(sim, "robust2025")
    e_new.append(e)
    used = r["fit_info"]["control_used"]
check("the control is used when it is a valid control", used)
check("robust pipeline recovers the true dF/F within 1 % (RMS)", np.mean(e_new) < 0.01,
      f"robust {np.mean(e_new):.4f} vs Douglass {np.mean(e_old):.4f}")
check("robust pipeline is at least as accurate as the old default", np.mean(e_new) <= np.mean(e_old) * 1.05)

print("\n=== control that moves AGAINST the signal (not isosbestic) ===")
sim = simulate(seed=3, leak=-0.4, motion_amp=0.0)
r, e = run(sim, "robust2025")
check("control rejected", r["fit_info"]["control_used"] is False and any("NOT used" in w for w in r["warnings"]),
      f"agreement r = {r['fit_info']['control_agreement_r']:.2f}")
r_forced, e_forced = run(sim, "robust2025", control_use="always")
check("...and the result is closer to the truth than when the control is forced in", e < e_forced,
      f"rejected {e:.4f} vs forced {e_forced:.4f}")
r_never, _ = run(simulate(seed=1), "robust2025", control_use="never")
check("'never' switches the control off", r_never["fit_info"]["control_used"] is False)

print("\n=== short recording with a sustained response (food to a fasted AgRP mouse) ===")
sim = simulate(T=300, seed=5, step=(60.0, -0.40), leak=-0.03, transients=False, motion_amp=0.01)
t, sig, ctl, true, fs = sim
for pl in ("robust2025", "douglass2025"):
    r, _ = run(sim, pl)
    d = r["dFF"]
    pre, post = np.nanmean(d[(t > 30) & (t < 55)]), np.nanmean(d[(t > 150) & (t < 280)])
    change = (1 + post) / (1 + pre) - 1
    if pl == "robust2025":
        check("a 40 % sustained fall is not absorbed as bleaching", abs(change - (-0.40)) < 0.03, f"{change:+.3f}")

print("\n=== slow trends opposed, fast artefacts shared ===")
t, sig, ctl, true, fs = simulate(T=400, seed=7, bleach=(0.10, 0.0), motion_amp=0.06)
ctl = ctl * (1 + 0.06 * t / t[-1])                       # control drifts UP while the signal bleaches
sim = (t, sig, ctl, true, fs)
r, e = run(sim, "robust2025")
r_off, e_off = run(sim, "robust2025", control_use="never")
fi = r["fit_info"]
check("control rejected as a whole, but its fast part still removes movement",
      fi["control_used"] is False and (fi.get("fast_correction_slope") or 0) > 0 and e < e_off,
      f"error {e:.4f} with fast correction vs {e_off:.4f} without")

print("\n=== background (camera dark level) ===")
sim0 = simulate(seed=2)
simb = simulate(seed=2, bg=6.0)
_, e_bg_ignored = run(simb, "robust2025")
_, e_bg_removed = run(simb, "robust2025", signal_background=6.0, control_background=6.0)
check("an unsubtracted background shrinks dF/F; subtracting it restores it",
      e_bg_removed < 0.01 < e_bg_ignored, f"removed {e_bg_removed:.4f}, ignored {e_bg_ignored:.4f}")
r, _ = run(simb, "robust2025", signal_background=60.0, control_background=60.0)
check("a background above the signal is reported", any("too high" in w for w in r["warnings"]))

print("\n=== fitting ===")
x = np.linspace(1, 2, 400)
y = 2 * x + 0.5
y2 = y.copy()
y2[300:] += 5
a, b = pc._bisquare_fit(x, y2)
check("bisquare fit ignores a block of outliers", abs(a - 2) < 1e-6 and abs(b - 0.5) < 1e-6, f"{a:.4f}, {b:.4f}")
a_ols, _ = pc._ols_fit(x, y2)
check("...which pull an ordinary fit away", abs(a_ols - 2) > 1)
rng = np.random.default_rng(0)
s_ = 1 + 0.1 * np.sin(np.arange(4000) / 50.0) + 0.01 * rng.standard_normal(4000)
check("agreement r: positive for shared changes, negative for opposite",
      pc.control_agreement(s_, s_ * 0.5 + 0.001 * rng.standard_normal(4000), 20)[0] > 0.8
      and pc.control_agreement(2 - s_, s_, 20)[0] < -0.8)

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
