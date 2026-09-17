"""Validate lickometer CSV auto-detection against many layouts."""
import io
import numpy as np
import pandas as pd
import lickometer as lk

rng = np.random.default_rng(7)
fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


def synth_licks(n_bouts=25, licks_per_bout=30, rate=7.0, gap=120.0, t0=50.0):
    """Ground-truth licking: bouts of ~7 Hz separated by minutes."""
    t, cur = [], t0
    for _ in range(n_bouts):
        for i in range(licks_per_bout):
            t.append(cur)
            cur += 1.0 / rate
        cur += gap
    return np.array(t)


TRUE = synth_licks()
N_TRUE = len(TRUE)
DUR = TRUE[-1] + 60
print(f"ground truth: {N_TRUE} licks, 25 bouts, 7 Hz, {DUR/3600:.2f} h span\n")


print("=== 1. REAL FILE: LIQ HD raw export ===")
r = lk.load_licks_csv(
    "/mnt/user-data/uploads/LIQHD_pup_removal_longitudinal_8135X_20260916_112521_raw.csv")
check("format identified", r.fmt == "LIQ HD raw export", r.fmt)
check("used the clock, not the corrupt counter", r.time_column == "pc_clock",
      r.time_column)
check("duration is plausible (23.4 h, not 9 years)", 23 < r.duration_sec / 3600 < 24,
      f"{r.duration_sec/3600:.2f} h")
check("split per bottle", list(r.sources) == ["bottle 2"], str(list(r.sources)))
check("warned about the disagreeing time columns",
      any("disagree" in w for w in r.warnings))
check("licks found", r.n_licks > 10000, str(r.n_licks))


print("\n=== 2. one timestamp per lick, plain seconds ===")
df = pd.DataFrame({"lick_time": TRUE})
r = lk.load_licks_csv(df)
check("all licks recovered", r.n_licks == N_TRUE, f"{r.n_licks}/{N_TRUE}")
check("absolute times preserved (not rebased to first lick)",
      np.allclose(r.combined, TRUE))


print("\n=== 3. timestamps in milliseconds (unit inferred) ===")
df = pd.DataFrame({"t_ms": TRUE * 1000.0})
r = lk.load_licks_csv(df)
check("all licks recovered", r.n_licks == N_TRUE, f"{r.n_licks}/{N_TRUE}")
check("converted to seconds", abs(r.duration_sec - TRUE[-1]) < 0.01,
      f"{r.duration_sec:.1f}s")


print("\n=== 4. timestamps in minutes ===")
df = pd.DataFrame({"time_min": TRUE / 60.0})
r = lk.load_licks_csv(df)
check("all licks recovered", r.n_licks == N_TRUE, f"{r.n_licks}/{N_TRUE}")
check("converted to seconds", abs(r.duration_sec - TRUE[-1]) < 0.5,
      f"{r.duration_sec:.1f}s")


print("\n=== 5. clock timestamps only (no elapsed column) ===")
base = pd.Timestamp("2026-09-16 09:00:00")
df = pd.DataFrame({"pc_clock": [base + pd.Timedelta(seconds=float(x)) for x in TRUE]})
r = lk.load_licks_csv(df)
check("all licks recovered", r.n_licks == N_TRUE, f"{r.n_licks}/{N_TRUE}")
check("start clock captured", r.start_clock is not None)
check("times relative to first timestamp",
      np.max(np.abs(r.combined - (TRUE - TRUE[0]))) < 0.002)


print("\n=== 6. time + single 0/1 state column (sampled at 100 Hz) ===")
fs = 100.0
grid = np.arange(0, DUR, 1 / fs)
state = np.zeros(grid.size, np.int8)
state[np.searchsorted(grid, TRUE)] = 1          # 1-sample pulse per lick
df = pd.DataFrame({"time_sec": grid, "lick": state})
r = lk.load_licks_csv(df)
check("format identified", "state" in r.fmt, r.fmt)
check("all licks recovered", r.n_licks == N_TRUE, f"{r.n_licks}/{N_TRUE}")
check("timing accurate to one sample",
      np.max(np.abs(r.combined - TRUE)) < 1.5 / fs,
      f"max err {np.max(np.abs(r.combined - TRUE))*1000:.1f} ms")


print("\n=== 7. multi-bottle: time + several state columns ===")
t2 = TRUE + 0.037
s1 = np.zeros(grid.size, np.int8); s1[np.searchsorted(grid, TRUE)] = 1
s2 = np.zeros(grid.size, np.int8); s2[np.searchsorted(grid, t2[t2 < DUR])] = 1
df = pd.DataFrame({"time_sec": grid, "lick_left": s1, "lick_right": s2})
r = lk.load_licks_csv(df)
check("two bottles found", len(r.sources) == 2, str(list(r.sources)))
check("left count right", r.sources["lick_left"].size == N_TRUE,
      str(r.sources["lick_left"].size))
check("combined pools both", r.n_licks == r.sources["lick_left"].size +
      r.sources["lick_right"].size)


print("\n=== 8. multi-bottle: long format with a channel column ===")
rows = []
for ch, shift in ((0, 0.0), (1, 0.05), (4, 0.11)):
    for x in TRUE:
        rows.append({"time_sec": x + shift, "channel": ch})
df = pd.DataFrame(rows)
r = lk.load_licks_csv(df)
check("three bottles found", len(r.sources) == 3, str(list(r.sources)))
check("labels name the channel", set(r.sources) == {"bottle 0", "bottle 1", "bottle 4"},
      str(set(r.sources)))
check("each has the right count",
      all(v.size == N_TRUE for v in r.sources.values()),
      str({k: v.size for k, v in r.sources.items()}))


print("\n=== 9. per-bin counts (e.g. a LIQ HD hourly file) ===")
edges = np.arange(0, DUR + 60, 60.0)
counts, _ = np.histogram(TRUE, bins=edges)
df = pd.DataFrame({"time_sec": edges[:-1], "licks": counts})
r = lk.load_licks_csv(df)
check("format identified", r.fmt == "per-bin lick counts", r.fmt)
check("total count preserved", r.n_licks == N_TRUE, f"{r.n_licks}/{N_TRUE}")
check("warned that microstructure is not real",
      any("microstructure" in w for w in r.warnings))


print("\n=== 10. awkward but legal files ===")
# pandas index column written out
df = pd.DataFrame({"lick_time": TRUE})
buf = io.StringIO(); df.to_csv(buf); buf.seek(0)
r = lk.load_licks_csv(buf)
check("unnamed index column ignored", r.n_licks == N_TRUE and "lick_time" in r.time_column,
      f"n={r.n_licks} col={r.time_column}")

# extra metadata columns that must not be mistaken for a time axis
df = pd.DataFrame({"subject": ["M1"] * N_TRUE, "lick_time": TRUE,
                   "peak": rng.integers(3, 40, N_TRUE)})
r = lk.load_licks_csv(df)
check("metadata columns ignored", r.n_licks == N_TRUE and r.time_column == "lick_time",
      f"n={r.n_licks} col={r.time_column}")

# unsorted rows
shuf = TRUE.copy(); rng.shuffle(shuf)
r = lk.load_licks_csv(pd.DataFrame({"lick_time": shuf}))
check("unsorted rows sorted", np.allclose(r.combined, TRUE))

# state column that starts already high
st = np.zeros(1000, np.int8); st[0:5] = 1; st[500:505] = 1
r = lk.load_licks_csv(pd.DataFrame({"time_sec": np.arange(1000) / 100.0, "lick": st}))
check("leading-high state counted", r.n_licks == 2, str(r.n_licks))


print("\n=== 11. debounce ===")
doubled = np.sort(np.concatenate([TRUE, TRUE + 0.004]))   # sensor double-counts
r = lk.load_licks_csv(pd.DataFrame({"lick_time": doubled}), min_inter_lick_sec=0.05)
check("duplicate contacts collapsed", r.n_licks == N_TRUE, f"{r.n_licks}/{N_TRUE}")
r0 = lk.load_licks_csv(pd.DataFrame({"lick_time": doubled}), min_inter_lick_sec=0.0)
check("debounce off keeps both", r0.n_licks == 2 * N_TRUE, str(r0.n_licks))
check("warns about impossible lick rate", any("faster than a mouse" in w
                                              for w in r0.warnings))


print("\n=== 12. failure modes are explicit, not silent ===")
try:
    lk.load_licks_csv(pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]}))
    check("no-time-column raises", False)
except ValueError as e:
    check("no-time-column raises with a clear message", "time column" in str(e).lower(),
          str(e)[:60])
try:
    lk.load_licks_csv(pd.DataFrame(columns=["t"]))
    check("empty file raises", False)
except ValueError as e:
    check("empty file raises", "no rows" in str(e).lower(), str(e)[:50])
# a LIQ HD file whose lick rows were filtered out
bad = pd.DataFrame({"record_type": ["session_start", "session_end"],
                    "pc_clock": ["2026-01-01T00:00:00", "2026-01-02T00:00:00"],
                    "channel": [0, 0]})
try:
    lk.load_licks_csv(bad)
    check("LIQ HD file with no licks raises", False)
except ValueError as e:
    check("LIQ HD file with no licks raises clearly", "no rows" in str(e).lower(),
          str(e)[:70])


print("\n=== 13. bout detection on the recovered times ===")
r = lk.load_licks_csv(pd.DataFrame({"lick_time": TRUE}))
b = lk.detect_bouts(r.combined, inter_bout_sec=1.0, min_licks_per_bout=3)
check("25 bouts", len(b) == 25, str(len(b)))
check("30 licks per bout", abs(b.n_licks.mean() - 30) < 0.01, f"{b.n_licks.mean():.2f}")
check("7 Hz within bout", abs(b.lick_rate_hz.mean() - 7.0) < 0.05,
      f"{b.lick_rate_hz.mean():.3f}")


print("\n=== 14. backward compatibility with app.py's existing calls ===")
df = pd.DataFrame({"lick_time": TRUE})
t, col = lk.licks_from_timestamp_csv(df, time_column="lick_time", unit="s",
                                     min_inter_lick_sec=0.05)
check("licks_from_timestamp_csv still works", t.size == N_TRUE, str(t.size))
# and now also copes with a clock column, which it used to return 0 licks for
dfc = pd.DataFrame({"pc_clock": [base + pd.Timedelta(seconds=float(x)) for x in TRUE]})
t2_, _ = lk.licks_from_timestamp_csv(dfc, time_column="pc_clock")
check("licks_from_timestamp_csv now handles clock strings", t2_.size == N_TRUE,
      str(t2_.size))
df2 = pd.DataFrame({"time_sec": grid, "lick": state})
t3 = lk.licks_from_state_csv(df2, "time_sec", "lick")
check("licks_from_state_csv still works", t3.size == N_TRUE, str(t3.size))
check("licks_from_digital still works",
      lk.licks_from_digital(state, fs).size == N_TRUE)
check("bout_summary_table still works",
      lk.bout_summary_table(b, r.combined, DUR)["Bouts"].iloc[0] == 25)


print("\n=== 15. plotting ===")
r = lk.load_licks_csv(pd.DataFrame({"time_sec": grid, "lick_left": s1,
                                    "lick_right": s2}))
fig = lk.plot_licks(r, bouts=b, title="test")
check("plot_licks builds a figure (2 bottles x raster+rate, plus bouts)",
      len(fig.data) == 5, f"{len(fig.data)} traces")
check("bouts drawn as ONE trace, not 25 layout shapes",
      len(fig.layout.shapes) == 0 and any("bout" in str(d.name) for d in fig.data))
sig_t = np.arange(0, DUR, 0.1)
sig_v = np.sin(sig_t / 300.0)
fig2 = lk.plot_licks_with_signal(sig_t, sig_v, r)
check("plot_licks_with_signal builds a figure", len(fig2.data) == 5,
      f"{len(fig2.data)} traces")
mat, taxis, kept = lk.peri_event_matrix(sig_t, sig_v, b.onset_sec.to_numpy(),
                                        pre_sec=10, post_sec=20)
fig3 = lk.plot_peri_event(mat, taxis)
check("plot_peri_event builds a figure", len(fig3.data) == 3, f"{len(fig3.data)}")
check("figures render to HTML", len(fig.to_html()) > 1000)
# big raster must be decimated, not dropped
big = lk.load_licks_csv(pd.DataFrame({"lick_time": np.arange(200000) * 0.3}))
fb = lk.plot_licks(big, max_raster_points=5000)
check("huge raster decimated for the browser", len(fb.data[0].x) <= 5000,
      f"{len(fb.data[0].x)} points drawn")
check("but the rate trace uses every lick", np.max(fb.data[1].y) > 3,
      f"peak {np.max(fb.data[1].y):.2f} Hz")

print("\n" + "=" * 62)
print("RESULT: " + ("ALL TESTS PASSED" if not fails
                    else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
