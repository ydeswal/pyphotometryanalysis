"""Tests for Plexon import, recordings without clock time in the AI, AI graphs, and AM/PM parsing."""
import io

import numpy as np
import pandas as pd

import plexon
import graph_tools as gt

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(1)
fs = 30.0
t = np.arange(0, 300, 1 / fs)
drop = 1 - 0.2 * (t > 125) * (1 - np.exp(-np.clip(t - 125, 0, None) / 5))
f465 = 17.5 * drop * (1 + 0.002 * rng.standard_normal(t.size))
f410 = 19.0 * (1 + 0.002 * rng.standard_normal(t.size))
f560 = 15.0 + 0.001 * rng.standard_normal(t.size)
rows = ["Frame,timestamp,RawF_465_F1, F0_465_F1, R_465_F1,RawF_410_F1, F0_410_F1, R_410_F1,RawF_560_F1, F0_560_F1, R_560_F1"]
for i in range(t.size):
    rows.append(f"{i + 1},{t[i]:.5f},{f465[i]:.5f},17.5,{f465[i] / 17.5 - 1:.5f},{f410[i]:.5f},19.0,0,"
                f"{f560[i]:.5f},15.0,0")
csv = "\n".join(rows)
ev_csv = 'Event,"Start Time (s)","End Time (s)"\n"drop",120.0,120.13\n"eating",200.5,260.2\n'

print("=== detection ===")
df = pd.read_csv(io.StringIO(csv))
check("photometry file detected", plexon.looks_like_photometry(df.columns))
check("channels per fibre", plexon.channels(df.columns) == {"F1": ["465", "410", "560"]})
check("default pair is 465 / 410", plexon.default_pair(["465", "410", "560"]) == ("465", "410"))
ev = pd.read_csv(io.StringIO(ev_csv))
check("event file detected", plexon.looks_like_events(ev.columns) and not plexon.looks_like_photometry(ev.columns))
check("a pyPhotometry CSV is not Plexon", not plexon.looks_like_photometry(["time_sec", "BLA_sig", "BLA_iso"]))

print("\n=== conversion ===")
two, info = plexon.to_two_channel(df, "BLA")
check("two-channel table", list(two.columns[:3]) == ["time_sec", "BLA_sig_raw", "BLA_uv_raw"]
      and np.allclose(two["BLA_sig_raw"], np.round(f465, 5)), str(list(two.columns)))
check("Plexon's own R kept", "plexon_R_465" in two.columns)
check("frame rate", abs(info["plexon_fs"] - 30) < 0.01 and info["plexon_frame_gaps"] == 0)
check("notes mention no clock time and 410 nm", any("not the clock time" in n for n in plexon.notes_for(info))
      and any("410" in n for n in plexon.notes_for(info)))
try:
    plexon.to_two_channel(df, "BLA", fibre="F2")
    check("unknown fibre raises", False)
except ValueError:
    check("unknown fibre raises", True)

print("\n=== events ===")
e = plexon.read_events(io.StringIO(ev_csv))
check("events read with start and end", e[0] == {"label": "drop", "start": 120.0, "end": 120.13}
      and e[1]["label"] == "eating" and abs(e[1]["end"] - 260.2) < 1e-9)
m = plexon.match_events(["8786N-events.csv", "8781R=events.csv", "8784Revents.csv", "8771R2-events.csv",
                         "9999-events.csv"], ["8786N", "8781R", "8784-R", "8771R", "8771R2"])
check("event files matched to recordings by name",
      m == {"8786N-events.csv": "8786N", "8781R=events.csv": "8781R", "8784Revents.csv": "8784-R",
            "8771R2-events.csv": "8771R2", "9999-events.csv": None}, str(m))

print("\n=== through the app pipeline and the AI ===")
import photometry_core as pc  # noqa: E402
res = pc.process_photometry(two["BLA_sig_raw"].to_numpy(), two["BLA_uv_raw"].to_numpy(), info["plexon_fs"])
out = pd.DataFrame({"time_sec": two["time_sec"], "BLA_sig_raw": two["BLA_sig_raw"], "BLA_uv_raw": two["BLA_uv_raw"],
                    "BLA_sig_filt": res["sig_filt"], "BLA_uv_filt": res["ctl_filt"], "BLA_dFF": res["dFF"],
                    "BLA_dFF_percent": 100 * res["dFF"], "BLA_z_dFF": res["z_dFF"]})
settings = dict(info, fit_R2=res["fit_info"].get("r2"))
check("pipeline runs on Plexon data (30 Hz)", np.isfinite(out["BLA_dFF_percent"]).mean() > 0.99)
import ai_chat as ac  # noqa: E402
ctx = ac.DataContext([{"name": "8786N", "df": out, "settings": settings}], None, "BLA",
                     events={"8786N": [(120.0, "drop"), (200.5, "eating")]})
check("no clock: times written as recording time", ac.DataContext.tfmt(ctx.clock_start(ctx.results[0])
                                                                        + pd.Timedelta(seconds=75),
                                                                        ctx.clock_start(ctx.results[0]))
      == "rec 00:01:15.0")
r = ac.run_tool(ctx, "get_signal_summary", {"recording": "8786N", "start": "rec 00:00:10", "end": "rec 00:01:00"})
check("signal summary works without clock time", "error" not in r and "rec 00:00:10" in str(r.get("window")),
      str(r)[:200])
r = ac.run_tool(ctx, "event_signal_check", {"event": "drop", "direction": "either"})
check("signal check on Plexon data", r.get("grades_best_first", [{}])[0].get("grade") in
      ("Excellent", "Very good"), str(r)[:300])
for kind in ac.GRAPH_KINDS[:4]:
    ctx.graphs = ctx.graphs[-1:]
    r = ac.run_tool(ctx, "show_graph", {"kind": kind, "event": "drop", "recordings": ["8786N"]})
    check(f"show_graph {kind}", "error" not in r, str(r)[:200])
r = ac.run_tool(ctx, "show_graph", {"kind": "signal"})
check("graphs recorded for the answer", len(ctx.graphs) == 3, str(len(ctx.graphs)))
r = ac.run_tool(ctx, "show_graph", {"kind": "signal"})
check("at most three graphs per answer", "error" in r)
fig, what = ac.build_figure(ctx, {"kind": "signal", "recordings": ["8786N"], "start": "rec 00:01:00",
                                  "end": "rec 00:03:00", "signal": "dff"})
check("signal graph has data and the event line inside the window", len(fig.data) == 1
      and len(fig.layout.shapes) == 1, what)
spec = ac.auto_graph([{"tool": "get_signal_summary", "args": {"recording": "8786N", "start": "rec 00:01:00"},
                       "result": {"mean": 1}}])
check("automatic graph from a signal question", spec and spec["kind"] == "signal" and spec["recordings"] ==
      ["8786N"])
spec = ac.auto_graph([{"tool": "event_signal_check", "args": {"event": "drop"}, "result": {}}])
check("automatic graph from a signal check", spec and spec["kind"] == "signal_check")

print("\n=== Plexon grading: rise during bouts / noise ===")
import signal_check as sc  # noqa: E402
tt = np.arange(0, 400, 1 / 30)
bouts_ = [(150.0, 190.0), (280.0, 320.0)]
bump = sum(((tt >= a) & (tt < b)) * (1 - np.exp(-(tt - a) / 4)) * np.exp(-np.clip(tt - b, 0, None) / 8)
           + ((tt >= b) * np.exp(-(tt - b) / 8)) for a, b in bouts_)
for amp, noise, want in ((1.0, 0.03, "Excellent"), (0.3, 0.03, "Good"), (0.05, 0.03, "No response")):
    y = amp * bump + noise * rng.standard_normal(tt.size)
    a = sc.assess_bouts(tt, y, bouts_, direction=sc.DIRECTIONS[2], label="eating")
    check(f"rise {amp} over noise {noise}: {want}", a["tier"] == want, f"{a['tier']} score {a['score']:.1f}")
a = sc.assess_bouts(tt, 0.03 * rng.standard_normal(tt.size), [(150.0, None)])
check("event without an end is scored over the next 30 s", a["response"] == (0.0, 30.0) and a["bouts"] == 1)
y = bump + 0.03 * rng.standard_normal(tt.size)
a = sc.assess_bouts(tt, y, bouts_, iso=5 + 0.8 * bump + 0.03 * rng.standard_normal(tt.size))
check("control rising with the signal: flagged", "410 moves too" in a["flags"], str(a["flags"]))

print("\n=== AM/PM lights ===")
check("parse 12-hour and 24-hour times", [gt.parse_tod(x) for x in ("7:00 PM", "7pm", "19:00", "6:30 am", "12 am",
                                                                    "12:15 PM")] == [19, 19, 19, 6.5, 0, 12.25])
check("reject nonsense", gt.parse_tod("13pm") is None and gt.parse_tod("abc") is None)
check("format", gt.fmt_tod(19.5) == "7:30 PM" and gt.fmt_tod(0) == "12:00 AM")

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
