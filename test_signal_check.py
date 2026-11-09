"""Tests for signal_check, event parsing/alignment in the group view, and the AI event tools."""
import numpy as np
import pandas as pd

import signal_check as sc
import group_analysis as ga

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(7)
fs = 20.0
t = np.arange(0, 240, 1 / fs)


def mouse(drop=0.3, iso_move=0.0, noise=0.01, tau=8.0, bleach=0.0, onset=65.0):
    """Raw 465 / 405 volts: 465 falls by `drop` (fraction) after `onset` s; 405 changes by `iso_move`."""
    step = (t >= onset) * (1 - np.exp(-np.clip(t - onset, 0, None) / tau))
    b = 1 - bleach * t / t[-1]
    sig = 1.5 * b * (1 - drop * step) * (1 + noise * rng.standard_normal(t.size))
    iso = 0.75 * b * (1 + iso_move * step) * (1 + noise * rng.standard_normal(t.size))
    return sig, iso


print("=== grades ===")
cases = {"excellent": (0.40, 0.015), "very good": (0.20, 0.01), "good": (0.12, 0.005), "ok": (0.07, 0.0),
         "weak": (0.03, 0.0), "none": (0.0, 0.0)}
want = {"excellent": "Excellent", "very good": "Very good", "good": "Good", "ok": "OK", "weak": "Weak",
        "none": "No response"}
for k, (d, iso) in cases.items():
    s, u = mouse(d, iso, noise=0.004)
    a = sc.assess(t, s, u, 60.0)
    check(f"{k}: grade {want[k]}", a["tier"] == want[k], f"{a['tier']} {a['change']:+.1f}%")
s, u = mouse(0.4, 0.015)
a = sc.assess(t, s, u, 60.0)
check("real calcium fall: 405 rises slightly, noted as expected", a["change_405"] > 0 and
      any("other way" in n for n in a["notes"]), str(a["notes"]))
check("time to half within seconds of the onset", 5 < a["t_half"] < 20, f"{a['t_half']:.1f}")

s, u = mouse(0.0, 0.0, bleach=0.08)
a = sc.assess(t, s, u, 60.0)
check("bleaching of both channels: ratio shows no response", a["tier"] == "No response", f"{a['change']:+.2f}")
a465 = sc.assess(t, s, u, 60.0, signal="465 raw")
check("...while raw 465 alone shows a fall", a465["change"] < -2, f"{a465['change']:+.2f}")

# movement-like: both channels fall together, 465 a bit more
s, u = mouse(0.12, -0.10)
a = sc.assess(t, s, u, 60.0, signal="465 raw")
check("405 moving the same way: flagged and graded down", "405 moves with 465" in a["flags"]
      and a["tier"] == "OK" and a["tier_before_checks"] == "Good", f"{a['tier']} / {a['tier_before_checks']}")

s, u = mouse(0.07, 0.0, noise=0.15)
a = sc.assess(t, s, u, 60.0)
check("change within the noise: graded down", "within noise" in a["flags"], f"snr {a['snr']:.1f}")

s, u = mouse(0.3)
a = sc.assess(t, s, u, 230.0)
check("event too close to the end: not testable", a["tier"] == "Not testable", a["notes"][0])
a = sc.assess(t, s, u, 60.0, direction=sc.DIRECTIONS[1])
check("expecting an increase: a fall is No response, and noted", a["tier"] == "No response" and
      any("opposite way" in n for n in a["notes"]))

A = [(f"m{i}", f"m{i}", sc.assess(t, *mouse(d), 60.0)) for i, d in enumerate([0.4, 0.2, 0.1, 0.05, 0.0])]
tab = sc.table(A)
check("table best first", list(tab["recording"]) == ["m0", "m1", "m2", "m3", "m4"], str(list(tab["recording"])))
G = sc.group_summary(A)
check("group summary: 5 recordings, mean negative", G["n"] == 5 and G["mean"] < -5)

print("\n=== sliding-mean recommendation ===")
check("short recording: off", sc.recommend_smoothing(7 * 60)[0] is False)
check("1 h: on, 1 min", sc.recommend_smoothing(3600)[:2] == (True, 60.0))
check("4 h: 5 min", sc.recommend_smoothing(4 * 3600)[1] == 300.0)
check("24 h and a week: 30 min (Douglass 2024)", sc.recommend_smoothing(86400)[1] == 1800.0
      and sc.recommend_smoothing(7 * 86400)[1] == 1800.0)

print("\n=== group alignment on events, % change ===")
m = ga.make_member("r1", "a", "All", t, np.full(t.size, 2.0), pd.Timestamp("2026-11-08 18:38:00"))
m["event_t"] = 60.0
check("event alignment puts the event at 0", ga.align_offset(m, "Time since an event") == -60.0)
m2 = dict(m, event_t=None)
check("no event: cannot align", ga.align_offset(m2, "Time since an event") is None)
bm = ga.baseline_mask(m, "Time since an event", 0.5)
check("event baseline = the 30 s before it", abs(t[bm].min() - 30) < 0.1 and t[bm].max() < 60)
y = np.r_[np.full(100, 2.0), np.full(100, 1.5)]
pc = ga.normalise(np.arange(200), y, "% change from baseline mean", np.arange(200) < 100)
check("% change from baseline", np.allclose(pc[:100], 0) and np.allclose(pc[100:], -25))

print("\n=== event time parsing (group view) ===")
import group_view as gv
check("recording time forms", [gv.parse_recording_time(x) for x in ("60", "1:00", "0:01:00", "1m30s", "-0:10")]
      == [60.0, 60.0, 60.0, 90.0, -10.0])
cs = pd.Timestamp("2026-11-08 20:25:54.2")
check("clock time 12 h", abs(gv.parse_clock_time("8:26:54.2 PM", cs) - 60.0) < 1e-6)
check("clock time 24 h", abs(gv.parse_clock_time("20:26:54.2", cs) - 60.0) < 1e-6)
check("clock time after midnight goes to the next day",
      abs(gv.parse_clock_time("12:05 AM", pd.Timestamp("2026-11-08 23:55")) - 600) < 1e-6)
ev = pd.DataFrame([{"event": "Food in", "recording": "All recordings", "time": "1:00", "time is": "Recording time"},
                   {"event": "Eat", "recording": "r2", "time": "8:28 PM", "time is": "Clock time"}])
res, bad = gv.resolve_events(ev, ["r1", "r2"], {"r1": cs, "r2": pd.Timestamp("2026-11-08 20:25:00")})
check("events resolved per recording", res["r1"] == [(60.0, "Food in")] and res["r2"] == [(60.0, "Food in"),
                                                                                      (180.0, "Eat")] and not bad,
      str(res))

print("\n=== AI tools ===")
import ai_chat as ac
results = []
for i, d in enumerate([0.4, 0.12, 0.0]):
    s, u = mouse(d, 0.01 if d else 0.0)
    df = pd.DataFrame({"time_sec": t, "ARC_sig_filt": s, "ARC_uv_filt": u,
                       "ARC_dFF_percent": (s / u / np.mean(s / u) - 1) * 100,
                       "clock_time": [str(pd.Timestamp("2026-11-08 18:00") + pd.Timedelta(minutes=10 * i)
                                          + pd.Timedelta(seconds=x)) for x in t]})
    results.append({"name": f"m{i}_sig", "df": df, "settings": {}})
ctx = ac.DataContext(results, None, "ARC", events={f"m{i}_sig": [(60.0, "Food in")] for i in range(3)})
r = ac.run_tool(ctx, "event_signal_check", {})
check("signal check tool: grades best first", [x["grade"] for x in r["grades_best_first"]] ==
      ["Excellent", "Good", "No response"], str(r.get("error") or [x["grade"] for x in r["grades_best_first"]]))
check("signal check tool: group data_check present", r.get("data_check", {}).get("level") in interpret_levels
      if (interpret_levels := ("good", "moderate", "uncertain", "not significant", "not testable")) else False,
      str(r.get("data_check")))
r2 = ac.run_tool(ctx, "event_signal_check", {"at": "rec 00:01:00", "recordings": ["m0_sig"]})
check("signal check at a given time for one recording", len(r2.get("grades_best_first", [])) == 1
      and r2["grades_best_first"][0]["grade"] == "Excellent", str(r2)[:300])
r3 = ac.run_tool(ctx, "list_events", {})
check("list_events gives clock and recording time", len(r3["events"]) == 3 and "PM" in r3["events"][0]["time"]
      and "rec 00:01:00" in r3["events"][0]["time"], str(r3["events"][0]))
check("catalogue lists the events", "EVENTS MARKED" in ac.catalogue(ctx))

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
