"""Tests for group_analysis and interpret (no data files needed)."""
import numpy as np
import pandas as pd

import group_analysis as ga
import interpret

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(3)
T = 6 * 3600


def member(name, animal, group, start, amp, dip=0.0, r2=0.5, day_shift=0):
    """6 h at 1 Hz, 12 meals of 3 pellets; 465 rises by `amp` after each
    meal; the 405 control dips by `dip`."""
    t = np.arange(0, T, 1.0)
    meals = np.linspace(900, T - 1200, 12) + rng.uniform(-120, 120, 12)
    y = 0.5 * rng.standard_normal(t.size)
    u = 0.5 * rng.standard_normal(t.size)
    for m in meals:
        e = np.exp(-np.clip(t - m, 0, None) / 90) * (t >= m)
        y += amp * e
        u -= dip * e
    cs = pd.Timestamp(start) + pd.Timedelta(days=day_shift)
    m = ga.make_member(name, animal, group, t, y, cs, fit_r2=r2, control=(t, (u - u.mean()) / u.std()))
    m["pellets"] = np.sort(np.concatenate([x + np.array([0.0, 20.0, 40.0]) for x in meals]))
    m["cov"] = [(0.0, float(T))]
    return m


A = [member(f"a{i}_d1", f"a{i}", "Day 1", "2026-11-04 15:57:30", amp) for i, amp in enumerate([3, 2, 4, 2.5])]
B = [member(f"a{i}_d7", f"a{i}", "Day 7", "2026-11-04 15:57:30", amp, day_shift=6)
     for i, amp in enumerate([1, 0.5, 1.5, 0.8])]
ART = member("a9_art", "a9", "Day 1", "2026-11-04 15:58:00", 0.0, dip=3.0)

print("=== alignment ===")
ref = ga.reference_clock(A + B)
check("reference clock is the earliest start, to the minute", ref == pd.Timestamp("2026-11-04 15:57"))
check("clock alignment keeps the 6-day gap", abs(ga.align_offset(B[0], ga.ALIGNMENTS[0], ref) - (6 * 86400 + 30)) < 1)
check("time of day overlays different days",
      ga.align_offset(A[0], ga.ALIGNMENTS[1]) == ga.align_offset(B[0], ga.ALIGNMENTS[1]))
check("first-pellet alignment puts the first pellet at 0",
      abs(ga.align_offset(A[0], ga.ALIGNMENTS[3]) + A[0]["pellets"][0]) < 1e-9)
S = ga.signal_matrix(A + B, ga.ALIGNMENTS[1], "z-score (whole recording)", 600, ref_clock=ref)
check("time of day: 8 recordings, all overlapping", S["M"].shape[0] == 8 and np.nanmax(
      ga.group_stat(S["M"])[3]) == 8, str(S["M"].shape))

print("\n=== normalisation and binning ===")
y = np.r_[np.zeros(50), np.ones(50) * 10, rng.standard_normal(900)]
z = ga.normalise(np.arange(y.size), y, "z-score (whole recording)")
check("z-score: mean 0, SD 1", abs(np.mean(z)) < 1e-9 and abs(np.std(z) - 1) < 1e-9)
rz = ga.normalise(np.arange(y.size), y, "Robust z (median / MAD)")
check("robust z: median 0", abs(np.median(rz)) < 1e-9)
bz = ga.normalise(np.arange(y.size), y, "z-score to a baseline period", np.arange(y.size) < 100)
check("baseline z uses only the baseline", abs(np.mean(bz[:100])) < 1e-9)
check("bin means", np.allclose(ga.bin_mean([0, 1, 2, 10, 11], [1, 2, 3, 10, 20], 0, 5, 3), [2, np.nan, 15],
                               equal_nan=True))
check("running mean ignores missing bins", np.allclose(ga.running_mean(np.array([1, np.nan, 3, 5.0]), 3)[[0, 2, 3]],
                                                       [1, 4, 4]))

print("\n=== group statistics ===")
M = np.array([[1.0, 2, np.nan], [3, 4, np.nan], [5, 9, 7]])
c, lo, hi, n = ga.group_stat(M, "Mean ± SEM")
check("mean ± SEM", np.allclose(c, [3, 5, 7]) and np.isclose(hi[0] - c[0], 2 / np.sqrt(3) * 1.0) and np.isnan(lo[2]),
      f"{c} {lo} {hi}")
check("n per bin", list(n) == [3, 3, 1])
c, lo, hi, n = ga.group_stat(M, "Median & IQR")
check("median & IQR", np.allclose(c[:2], [3, 4]) and np.isclose(lo[0], 2) and np.isclose(hi[0], 4))
c, lo, hi, n = ga.group_stat(np.tile(np.arange(5.0)[:, None], (1, 4)), "Mean ± 95% CI (bootstrap)")
check("bootstrap CI contains the mean and is inside the data range", np.all((lo <= 2) & (hi >= 2) & (lo >= 0)
                                                                             & (hi <= 4)))
per = ga.signal_matrix(A + B, ga.ALIGNMENTS[1], "As recorded", 600, ref_clock=ref, per_animal=True)
check("same animal in two groups stays separate per group", len(per["rows"]) == 8)
same = [dict(A[0], group="All"), dict(B[0], group="All")]
per = ga.signal_matrix(same, ga.ALIGNMENTS[1], "As recorded", 600, per_animal=True)
check("recordings of one animal in one group are averaged first", len(per["rows"]) == 1 and "mean of 2" in
      per["rows"][0])

print("\n=== FED ===")
F = ga.fed_matrix(A, ga.ALIGNMENTS[0], 1800, "Cumulative intake (g)", ref)
check("cumulative intake ends at 36 pellets x 20 mg", np.allclose(np.nanmax(F["M"], axis=1), 0.72))
F = ga.fed_matrix(A, ga.ALIGNMENTS[0], 1800, "Cumulative, % of own total", ref)
check("% of own total ends at 100", np.allclose(np.nanmax(F["M"], axis=1), 100))
F = ga.fed_matrix(A, ga.ALIGNMENTS[0], 1800, "Pellets per bin", ref)
check("pellets per bin add up", np.allclose(np.nansum(F["M"], axis=1), 36))
gap = dict(A[0], cov=[(0.0, 3600.0), (7200.0, float(T))])
F = ga.fed_matrix([gap], ga.ALIGNMENTS[2], 1800, "Pellets per bin")
check("bins without FED logging are missing, not zero", np.isnan(F["M"][0][2]) and np.isnan(F["M"][0][3]),
      str(F["M"][0][:5]))
F = ga.fed_matrix(A, ga.ALIGNMENTS[0], 1800, "Animals eating (%)", ref)
check("animals eating is a percentage", np.nanmax(F["M"]) == 100 and np.nanmin(F["M"]) == 0)
tab, summ = ga.meal_table(A, 60, 2, 7, 19)
check("meal table: 12 meals of 3 pellets each", list(tab["meals"]) == [12] * 4 and
      np.allclose(tab["meal size (pellets)"], 3))
check("group summary is mean ± SEM", "±" in str(summ.iloc[0]["pellets"]) or summ.iloc[0]["pellets"] == "36 ± 0",
      str(summ.iloc[0]["pellets"]))
check("dark-phase share computed", "% pellets in dark" in tab.columns)

print("\n=== meal-aligned group and the 405 rule ===")
R = [ga.member_meal_response(m, 300, 1, n_perm=500) for m in A]
G = ga.group_meal_response(R)
v = G["Day 1"]
check("responders detected; animals are the unit", v["n_animals"] == 4 and v["mean_delta"] > 1 and v["t_p"] < 0.05,
      f"{v['n_animals']} {v['mean_delta']:.2f} {v['t_p']:.3g}")
check("clean group is trusted", v["check"]["level"] in ("good", "moderate"), v["check"]["verdict"])
check("Wilcoxon floor reported for small n", np.isclose(v["wilcoxon_min_p"], 2 / 16))
r_art = ga.member_meal_response(ART, 300, 1, n_perm=500)
check("405-only animal: 465 does not respond", r_art["check"]["level"] == "not significant", r_art["check"]["verdict"])
R2 = [ga.member_meal_response(m, 300, 1, n_perm=500) for m in A] + [r_art]
r_art2 = dict(r_art, mean_delta=3.0, check=dict(r_art["check"], level="uncertain"))
G2 = ga.group_meal_response(R[:4] + [r_art2])
check("one animal with a moving 405 makes the group uncertain, named, clean subset re-tested",
      G2["Day 1"]["check"]["level"] == "uncertain" and "a9" in " ".join(G2["Day 1"]["check"]["reasons"])
      and any("flat 405 control only" in x for x in G2["Day 1"]["check"]["reasons"]),
      G2["Day 1"]["check"]["verdict"])
RB = [ga.member_meal_response(m, 300, 1, n_perm=300) for m in B]
c = ga.compare_groups(R + RB, "Day 1", "Day 7")
check("same animals on two days: paired comparison", c and c["design"].startswith("paired") and c["diff"] < 0,
      str(c))

print("\n=== trust rules ===")
a = interpret.assess_response(20, 0.3, 0.004, -0.04, -0.98, 0.0005, 0.009, unit="meals")
check("Q4 case: significant, 405 opposite and larger, low R² -> uncertain", a["level"] == "uncertain"
      and any("OPPOSITE" in x for x in a["reasons"]) and any("larger" in x for x in a["reasons"])
      and any("does not mean there was little motion" in x for x in a["reasons"]), a["verdict"])
a = interpret.assess_response(20, 0.3, 0.004, 0.0, 0.01, 0.6, 0.5)
check("clean case -> good", a["level"] == "good")
a = interpret.assess_response(20, 0.3, 0.2, 0.0, 0.01, 0.6, 0.5)
check("not significant", a["level"] == "not significant")
a = interpret.assess_response(5, 0.3, 0.01, 0.0, 0.01, 0.6, 0.5)
check("few events -> at most moderate", a["level"] == "moderate")
a = interpret.assess_response(20, 0.3, 0.01, 0.0, None, None, 0.5)
check("405 unknown -> at most moderate", a["level"] == "moderate")
check("overclaims caught when not 'good'", set(interpret.find_overclaims(
    "We are highly confident this is genuine; it rules out artifact and shows minimal motion.", ["uncertain"])) >=
      {"highly confident", "genuine", "rules out", "minimal motion"})
check("no false alarm on a careful answer", interpret.find_overclaims(
    "The signal rises, but the 405 control also changes, so this may not be neural.", ["uncertain"]) == [])
check("'proves' is never allowed", interpret.find_overclaims("This proves it.", ["good"]) == ["proves"])
notes = interpret.recording_notes({"fit_R2": 0.009, "clock_check": {"pc_drift_sec": 2.09}, "pc_timestamps": True})
check("R² note says low R² is not low motion", any("does NOT show" in n for n in notes))
check("drift note says it is corrected", any("already corrected" in n for n in notes))

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
