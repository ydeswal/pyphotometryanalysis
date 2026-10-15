"""Tests for graph_tools: decimation, peak ranking, removal, time formatting."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go

import graph_tools as gt

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


rng = np.random.default_rng(3)
fs = 1.0
t = np.arange(86_400) / fs                        # 24 h at 1 Hz
y = 0.3 * rng.standard_normal(t.size)
y[20_000] = 9.0                                    # one-sample artefact spike
y[50_000:50_300] += 4.0 * np.exp(-np.arange(300) / 80.0)   # broad transient
y[70_000] = -7.0                                   # deep trough

print("=== decimation ===")
idx = gt.minmax_indices(y, 12_000)
check("point count bounded", idx.size <= 12_000 * 1.3, f"{idx.size}")
check("global max kept", 20_000 in set(idx))
check("global min kept", 70_000 in set(idx))
check("indices sorted and in range", np.all(np.diff(idx) > 0) and idx[-1] < y.size)
yg = y.copy(); yg[30_000:31_000] = np.nan
idx2 = gt.minmax_indices(yg, 12_000)
check("NaN gap preserved (line not bridged)", np.isnan(yg[idx2]).any())
check("no decimation when short", gt.minmax_indices(y[:500], 12_000) is None)

fig = go.Figure([go.Scatter(x=t, y=y, mode="lines", customdata=np.c_[t, t]),
                 go.Scatter(x=t[:50], y=y[:50], mode="markers"),
                 go.Scatter(x=[0, 0, 1, 1, np.nan], y=[0, 1, 1, 0, np.nan], fill="toself")])
gt.lighten_figure(fig, 12_000)
check("long line decimated", len(fig.data[0].x) < 15_000, f"{len(fig.data[0].x)}")
check("customdata decimated with it", len(fig.data[0].customdata) == len(fig.data[0].x))
check("marker trace untouched", len(fig.data[1].x) == 50)
check("filled shape untouched", len(fig.data[2].x) == 5)
check("max still drawn", np.nanmax(fig.data[0].y) == 9.0)

print("\n=== peaks / troughs ===")
hi = gt.rank_extremes(t, y, n=5, min_sep_sec=60, highest=True)
lo = gt.rank_extremes(t, y, n=5, min_sep_sec=60, highest=False)
check("highest = the spike", hi["idx"].iloc[0] == 20_000, f"{hi['idx'].iloc[0]}")
check("second = the transient", abs(hi["time_sec"].iloc[1] - 50_000) < 60, f"{hi['time_sec'].iloc[1]}")
check("events are distinct (>= 60 s apart)",
      np.all(np.diff(np.sort(hi["time_sec"].to_numpy())) >= 60))
check("lowest = the trough", lo["idx"].iloc[0] == 70_000)
check("rank column 1..n", list(hi["rank"]) == [1, 2, 3, 4, 5])

base = float(np.median(y))
a, b = gt.excursion_bounds(t, y, 20_000, base, 60)
check("spike removal takes only the spike", b - a <= 2, f"{a}-{b}")
a, b = gt.excursion_bounds(t, y, 50_000, base, 600)
check("transient removal takes the whole transient", a <= 50_000 and b - a >= 100, f"{a}-{b}")
a, b = gt.excursion_bounds(t, y, 50_000, base, 30)
check("removal capped by max half-width", b - a <= 60, f"{a}-{b}")
y2 = gt._blank(t, y, [gt.excursion_bounds(t, y, 20_000, base, 60)])
check("after removing the spike, the transient is the new ceiling",
      abs(t[np.nanargmax(y2)] - 50_000) < 60)

print("\n=== time formatting / ticks ===")
check("fmt_hms", gt.fmt_hms(3725.256) == "01:02:05.26", gt.fmt_hms(3725.256))
check("fmt_hms negative", gt.fmt_hms(-12.5) == "-00:00:12.50", gt.fmt_hms(-12.5))
cs = pd.Timestamp("2026-10-14 10:23:00.423")
check("fmt_clock", gt.fmt_clock(cs, 3600) == "2026-10-14 11:23:00.42", gt.fmt_clock(cs, 3600))
v, lab = gt.time_ticks(0, 86_400, cs)
check("clock ticks land on round clock times",
      all((cs + pd.Timedelta(seconds=float(x))).round("ms").minute == 0 for x in v), str(lab[:3]))
v, lab = gt.time_ticks(0, 7200)
check("elapsed ticks", lab[0] == "00:00:00", str(lab[:3]))
df = pd.DataFrame({"time_sec": [100.0, 101.0], "clock_time": ["2026-10-14 10:24:40.423", "x"]})
check("clock start from a cropped frame",
      gt.clock_start_of(df) == pd.Timestamp("2026-10-14 10:23:00.423"))

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
