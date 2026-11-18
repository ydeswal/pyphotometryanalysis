"""Hand-checked numbers for the meal list / feeding pattern and the group baseline normalisation."""
import numpy as np
import pandas as pd

import baseline as bl
import group_analysis as ga

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


def close(a, b, tol=1e-9):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.allclose(a, b, atol=tol, equal_nan=True)


cs = pd.Timestamp("2026-11-17 09:58:00")
t = np.arange(0, 1800.0, 1.0)
m = ga.make_member("rec1", "m1", "All", t, np.zeros(t.size), cs, pellet_grams=0.02)
m["pellets"] = np.array([100, 120, 130, 500, 1000, 1010, 1030, 1100], float)

print("=== every meal, by hand (gap 60 s, at least 1 pellet) ===")
L = ga.meal_list([m], 60.0, 1, lights_on_h=22.0, lights_off_h=10.0)
check("four meals", list(L["pellets"]) == [3, 1, 3, 1], str(list(L["pellets"])))
check("onsets and offsets", close(L["onset_s"], [100, 500, 1000, 1100]) and close(L["offset_s"], [130, 500, 1030, 1100]))
check("durations (min)", close(L["duration_min"], [0.5, 0, 0.5, 0]))
check("grams = pellets x 20 mg", close(L["grams"], [0.06, 0.02, 0.06, 0.02]))
check("time since previous meal ended (min)", close(L["gap_before_min"], [np.nan, 370 / 60, 500 / 60, 70 / 60]),
      str(L["gap_before_min"].round(4).tolist()))
check("interval after (min)", close(L["interval_after_min"], [370 / 60, 500 / 60, 70 / 60, np.nan]))
check("satiety ratio = interval after / pellets",
      close(L["satiety_ratio"], [370 / 60 / 3, 500 / 60, 70 / 60 / 3, np.nan]))
check("eating rate in a meal = (pellets - 1) / duration", close(L["pellets_per_min"], [4, np.nan, 4, np.nan]))
check("clock time of onset", L["onset_clock"].iat[0] == pd.Timestamp("2026-11-17 09:59:40"))
check("dark phase 10 AM-10 PM: 9:59:40 is light, 10:06:20 is dark",
      list(L["dark"]) == [False, True, True, True], str(list(L["dark"])))

print("\n=== at least 2 pellets: singletons dropped, intervals between the kept meals ===")
L2 = ga.meal_list([m], 60.0, 2)
check("two meals", list(L2["pellets"]) == [3, 3])
check("gap before the 2nd = 1000 - 130 s", close(L2["gap_before_min"], [np.nan, 870 / 60]))
check("satiety of the 1st = 870 s / 3 pellets", close(L2["satiety_ratio"], [870 / 60 / 3, np.nan]))

print("\n=== the per-animal table agrees with the meal list ===")
tab, summ = ga.meal_table([m], 60.0, 1, 22.0, 10.0)
r = tab.iloc[0]
check("pellets and meals", r["pellets"] == 8 and r["meals"] == 4)
check("mean meal size", abs(r["meal size (pellets)"] - L["pellets"].mean()) < 1e-12)
check("mean duration (min)", abs(r["meal duration (min)"] - L["duration_min"].mean()) < 1e-12)
check("mean inter-meal interval (min)", abs(r["inter-meal interval (min)"] - L["gap_before_min"].mean()) < 1e-12)
check("mean satiety ratio", abs(r["satiety ratio (min/pellet)"] - L["satiety_ratio"].mean()) < 1e-12)
check("% pellets in meals = 100 with min 1 pellet", abs(r["% pellets in meals"] - 100) < 1e-12)
check("% pellets in dark (7 of 8 after 10:00 AM)", abs(r["% pellets in dark"] - 87.5) < 1e-9, str(r["% pellets in dark"]))
check("pellets per hour over the 0.5 h recording", abs(r["pellets / h"] - 8 / (1799 / 3600)) < 1e-9)

print("\n=== group baseline at a chosen moment ===")
rng = np.random.default_rng(0)
t = np.arange(0, 3600.0, 1.0)
members = []
for i, (sd, step) in enumerate(((1.0, 5.0), (2.0, 4.0))):
    y = 10 + sd * rng.standard_normal(t.size) + step * (t >= 1800)
    mm = ga.make_member(f"r{i}", f"a{i}", "All", t, y, cs + pd.Timedelta(seconds=60 * i))
    members.append(mm)
ref = ga.reference_clock(members)
end_x = 1800.0          # clock alignment: x = seconds since the first start; before both steps
for mm in members:
    bm = ga.baseline_mask(mm, "Clock time (same moment)", 10.0, end_x, ref)
    off = ga.align_offset(mm, "Clock time (same moment)", ref)
    xs = mm["t"][bm] + off
    check(f"{mm['name']}: baseline = the 10 min before the moment, on the shared axis",
          bm.sum() == 600 and xs.min() >= end_x - 600 and xs.max() < end_x)
    z = ga.normalise(mm["t"], mm["y"], "z-score to a baseline period", bm)
    check(f"{mm['name']}: z is 0 +/- 1 inside its baseline", abs(z[bm].mean()) < 1e-9 and abs(z[bm].std() - 1) < 1e-9)
S = ga.signal_matrix(members, "Clock time (same moment)", "z-score to a baseline period", 60.0, 1, 10.0, ref,
                     per_animal=False, baseline_end=end_x)
xb = S["x"]
inb = (xb > end_x - 600) & (xb < end_x)
check("binned group matrix: baseline bins average 0", abs(np.nanmean(S["M"][:, inb])) < 0.15,
      f"{np.nanmean(S['M'][:, inb]):.3f}")
post = (xb > end_x + 120) & (xb < end_x + 1200)
check("after the step: z = step / SD for each (5 and 2)",
      abs(np.nanmean(S["M"][0, post]) - 5) < 0.5 and abs(np.nanmean(S["M"][1, post]) - 2) < 0.3,
      f"{np.nanmean(S['M'][0, post]):.2f}, {np.nanmean(S['M'][1, post]):.2f}")
S0 = ga.signal_matrix(members, "Clock time (same moment)", "z-score to a baseline period", 60.0, 1, 10.0, ref,
                      per_animal=False)
check("a different baseline end gives a different graph", not np.allclose(S0["M"], S["M"], equal_nan=True))
off1 = ga.align_offset(members[1], "Clock time (same moment)", ref)
r1 = bl.peak_vs_baseline(members[1]["t"] + off1, members[1]["y"], end_x, 600.0, 600.0, "up", 30.0)
check("peak vs baseline on the shared axis (z about 2, % change about 40 % of 10)", r1["ok"] and 1.6 < r1["peak_z"] < 3.5,
      f"z {r1['peak_z']:.2f}")

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
