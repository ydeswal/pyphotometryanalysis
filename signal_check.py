#!/usr/bin/env python3
"""
signal_check.py
===============

How well does each recording respond to an event (food in the cage, an
injection, a stimulus)? Used to grade signal quality across animals, e.g. a
"signal check" of AgRP-GCaMP mice after a fast: AgRP neurons fall within
seconds of food being presented (Chen et al. 2015 Cell; Betley et al. 2015
Nature), so a mouse whose 465 signal does not fall is a likely surgical miss
(Deem et al. 2020 eLife used a >= 10 % fall in F/F on chow presentation as the
cut-off).

The numbers
-----------
For each recording, with the event at time 0:

  baseline     mean of the signal from `baseline[0]` to `baseline[1]` s
               (default -30 to -5 s: just before the event, skipping the
               seconds when the cage is being opened)
  change       mean of the signal from `response[0]` to `response[1]` s
               (default +30 to +120 s: long enough after the event to allow
               for the time the mouse needs to notice the food) as a % change
               from the baseline
  deepest      the most extreme 10-s mean between 0 and response[1], % change
  time to half time after the event until the signal first reaches half of
               the change (a step-like fall is fast; slow bleaching is not)
  405 change   the same % change of the raw 405 channel
  noise        SD of 1-s means in the baseline, in %; change / noise says
               whether the change stands out from the ongoing fluctuation

Signal used: by default the ratio 465 / 405 of the raw channels. Bleaching and
most light-path changes affect both channels by about the same factor, so the
ratio removes them, while a real calcium change moves 465 only. (The 405 nm
light is a few nm below GCaMP's true isosbestic point, ~410 nm, so a real
calcium FALL makes 405 RISE slightly; the ratio then shows the fall a little
more strongly, never less.)

Grades (on the change in the expected direction)
------------------------------------------------
  Excellent    >= 25 %
  Very good    15 - 25 %
  Good         10 - 15 %     (10 % = Deem et al. 2020 cut-off for a hit)
  OK           5 - 10 %      below the published cut-off; usable with care
  Weak         2 - 5 %
  No response  < 2 %, or the wrong direction

Two checks can lower a grade by one step:
  * the change is less than 3x the baseline noise (it does not stand out);
  * the 405 channel moves in the SAME direction as 465 by at least half as
    much (and >= 2 %): the change may be movement / light-path, not calcium.
A 405 change of >= 5 % in either direction is always flagged for a look.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

SIGNALS = ["465 ÷ 405 ratio (raw channels)", "465 raw", "dF/F (motion-corrected)"]
DIRECTIONS = ["Decrease (e.g. AgRP neurons when food appears)", "Increase", "Either direction"]
BASELINE = (-30.0, -5.0)
RESPONSE = (30.0, 120.0)
DEEP_WIN = 10.0
NOISE_RATIO = 3.0
ISO_SAME_FRACTION = 0.5
ISO_FLAG_PCT = 5.0

TIERS = [  # name, minimum change (%), colour
    ("Excellent", 25.0, "#047857"),
    ("Very good", 15.0, "#059669"),
    ("Good", 10.0, "#65a30d"),
    ("OK", 5.0, "#ca8a04"),
    ("Weak", 2.0, "#ea580c"),
    ("No response", -math.inf, "#b91c1c"),
]
TIER_NAMES = [t[0] for t in TIERS]
TIER_COLOR = {t[0]: t[2] for t in TIERS}
TIER_COLOR["Not testable"] = "#64748b"

CRITERIA_MD = """
**What is measured** (event at time 0, for every recording separately)

| Number | How it is computed | Why |
|---|---|---|
| **Change (%)** | mean signal from **{r0:+g} to {r1:+g} s** vs the mean from **{b0:+g} to {b1:+g} s**, as % change | the sustained response; the {r0:g} s gap gives the mouse time to notice the food |
| Deepest 10 s (%) | most extreme 10-s mean between 0 and {r1:+g} s | the full size of the response |
| Time to half (s) | first time the signal reaches half of the change | a real neural response is a step within seconds; bleaching is a slow slope |
| 465 / 405 change (%) | the same change for each raw channel | where the change comes from |
| Noise (%) | SD of 1-s means in the baseline | how much the signal moves anyway |
| Change ÷ noise | | does the change stand out? |

**Signal:** {signal}. The 465 ÷ 405 ratio cancels bleaching and light-path changes, which scale both channels, and
keeps calcium changes, which move 465 only.

**Grades** (change in the expected direction)

| Grade | Change | Meaning |
|---|---|---|
| 🟢 Excellent | ≥ 25 % | large, unmistakable response |
| 🟢 Very good | 15–25 % | clear response |
| 🟢 Good | 10–15 % | clear; passes the 10 % cut-off used by Deem et al. 2020 (*eLife*) to accept AgRP mice |
| 🟡 OK | 5–10 % | small; below the published cut-off, use with care |
| 🟠 Weak | 2–5 % | barely there |
| 🔴 No response | < 2 % or wrong direction | likely miss, poor expression, or the fibre is off target |

**A grade drops one step when**
* the change is less than **3 ×** the baseline noise (it does not stand out from the ongoing fluctuation), or
* the **405 channel moves the same way** as 465 by at least half as much (≥ 2 %): that pattern fits movement or a
  light-path change, not calcium.

A 405 change of 5 % or more in either direction is always flagged. A real calcium fall makes 405 *rise* slightly
(405 nm is a few nm below GCaMP's isosbestic point), so "465 down, 405 slightly up" is what a genuine AgRP
response looks like.

**Background.** AgRP neurons are highly active after a fast and fall within seconds of food being seen or smelled,
before eating starts (Chen et al. 2015 *Cell*; Betley et al. 2015 *Nature*; Mandelblat-Cerf et al. 2015 *eLife*).
Deem et al. 2020 (*eLife*) excluded overnight-fasted mice whose F/F fell by less than 10 % on chow presentation as
surgical misses. The grades above extend that single cut-off into steps so mice can be ranked. They are a quality
screen, not a statistical test: each mouse has one event.
"""


def _f(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return f"{x:+.{nd}f}" if math.isfinite(x) else "n/a"


def _mean(t, y, lo, hi):
    k = (t >= lo) & (t < hi) & np.isfinite(y)
    return (float(np.mean(y[k])) if k.sum() else float("nan")), int(k.sum())


def _roll(t, y, w):
    if t.size < 2:
        return y.copy()
    s = pd.Series(y, index=pd.to_timedelta(t - t[0], unit="s"))
    return s.rolling(pd.Timedelta(seconds=max(w, 1e-3)), center=True, min_periods=1).mean().to_numpy(float)


def pct_trace(t, y, event, baseline=BASELINE):
    """% change of y from its baseline mean (y must be positive, e.g. raw volts or a ratio)."""
    t = np.asarray(t, float) - float(event)
    y = np.asarray(y, float)
    b, _ = _mean(t, y, *baseline)
    if not math.isfinite(b) or b == 0:
        return t, np.full_like(y, np.nan)
    return t, (y / b - 1.0) * 100.0


def diff_trace(t, y, event, baseline=BASELINE):
    """Change of y from its baseline mean, in y's own units (for dF/F in %)."""
    t = np.asarray(t, float) - float(event)
    y = np.asarray(y, float)
    b, _ = _mean(t, y, *baseline)
    return t, y - b


def tier_of(score):
    if score is None or not math.isfinite(score):
        return "Not testable"
    for name, lo, _ in TIERS:
        if score >= lo:
            return name
    return TIERS[-1][0]


def _down(tier):
    if tier not in TIER_NAMES:
        return tier
    i = TIER_NAMES.index(tier)
    return TIER_NAMES[min(i + 1, len(TIER_NAMES) - 1)]


def assess(t, sig, iso, event, signal=SIGNALS[0], dff=None, baseline=BASELINE, response=RESPONSE,
           direction=DIRECTIONS[0], deep_win=DEEP_WIN):
    """
    Signal check of one recording around one event.

      t      seconds since the recording started
      sig    raw 465 (volts)          iso   raw 405 (volts)
      dff    dF/F in % (only needed when signal is the dF/F option)
      event  event time, seconds since the recording started

    Returns a dict of numbers, the grade and plain-language notes.
    """
    t = np.asarray(t, float)
    sig = np.asarray(sig, float)
    iso = np.asarray(iso, float) if iso is not None else None
    out = {"event_s": float(event), "signal": signal, "baseline": tuple(baseline), "response": tuple(response),
           "notes": [], "flags": []}
    tt = t - float(event)
    nb = int(((tt >= baseline[0]) & (tt < baseline[1])).sum())
    nr = int(((tt >= response[0]) & (tt < response[1])).sum())
    fs = 1.0 / np.nanmedian(np.diff(t)) if t.size > 2 else float("nan")
    need_b = 0.5 * (baseline[1] - baseline[0]) * fs if math.isfinite(fs) else 1
    need_r = 0.5 * (response[1] - response[0]) * fs if math.isfinite(fs) else 1
    if nb < max(5, need_b) or nr < max(5, need_r):
        out.update(tier="Not testable", score=float("nan"), change=float("nan"))
        why = []
        if nb < max(5, need_b):
            why.append(f"less than half of the baseline window ({baseline[0]:+g} to {baseline[1]:+g} s) is inside "
                       "the recording")
        if nr < max(5, need_r):
            why.append(f"less than half of the response window ({response[0]:+g} to {response[1]:+g} s) is inside "
                       "the recording")
        out["notes"].append("Not testable: " + "; ".join(why) + ".")
        return out

    # the trace being graded, as % change from baseline
    if signal.startswith("465 ÷ 405") and iso is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = sig / iso
        _, y = pct_trace(t, r, event, baseline)
    elif signal.startswith("dF/F") and dff is not None:
        _, y = diff_trace(t, np.asarray(dff, float), event, baseline)
    else:
        _, y = pct_trace(t, sig, event, baseline)
    _, s465 = pct_trace(t, sig, event, baseline)
    s405 = pct_trace(t, iso, event, baseline)[1] if iso is not None else None

    ch, _ = _mean(tt, y, *response)
    c465, _ = _mean(tt, s465, *response)
    c405 = _mean(tt, s405, *response)[0] if s405 is not None else float("nan")

    if direction.startswith("Decrease"):
        sign = -1.0
    elif direction.startswith("Increase"):
        sign = 1.0
    else:
        sign = -1.0 if ch < 0 else 1.0
    score = sign * ch

    sm = _roll(tt, y, deep_win)
    kw = (tt >= 0) & (tt <= response[1]) & np.isfinite(sm)
    deep, deep_at = float("nan"), float("nan")
    if kw.any():
        i = int(np.argmax(sign * sm[kw]))
        deep, deep_at = float(sm[kw][i]), float(tt[kw][i])

    one = _roll(tt, y, 1.0)
    kb = (tt >= baseline[0]) & (tt < baseline[1])
    noise = float(np.nanstd(one[kb])) if kb.sum() > 2 else float("nan")
    snr = score / noise if math.isfinite(noise) and noise > 0 else float("nan")

    t_half = float("nan")
    if score > 0:
        two = _roll(tt, y, 2.0)
        hit = np.where((tt >= 0) & (tt <= response[1]) & (sign * two >= score / 2.0))[0]
        if hit.size:
            t_half = float(tt[hit[0]])

    tier = tier_of(score)
    raw_tier = tier
    if tier not in ("No response", "Not testable"):
        if math.isfinite(snr) and snr < NOISE_RATIO:
            out["notes"].append(f"The change is only {snr:.1f}× the baseline noise (needs ≥ {NOISE_RATIO:g}×), so "
                                "the grade drops one step.")
            out["flags"].append("within noise")
            tier = _down(tier)
    iso_note = ""
    if math.isfinite(c405) and math.isfinite(c465):
        same = c405 * c465 > 0
        if same and abs(c405) >= ISO_SAME_FRACTION * abs(c465) and abs(c405) >= 2.0:
            out["flags"].append("405 moves with 465")
            iso_note = (f"405 changes the SAME way as 465 ({_f(c405)} % vs {_f(c465)} %): this fits movement or a "
                        "light-path change rather than calcium.")
            if tier not in ("No response", "Not testable"):
                tier = _down(tier)
                iso_note += " The grade drops one step."
        elif abs(c405) >= ISO_FLAG_PCT:
            out["flags"].append("405 moves ≥ 5 %")
            iso_note = f"405 changes by {_f(c405)} %; look at the raw traces before trusting this recording."
        elif (c465 < 0 < c405 or c465 > 0 > c405) and abs(c405) >= 0.5:
            iso_note = (f"405 moves slightly the other way ({_f(c405)} %), as expected for a real calcium change "
                        "(405 nm is a little below GCaMP's isosbestic point).")
        else:
            iso_note = f"405 is nearly flat ({_f(c405)} %)."
    if iso_note:
        out["notes"].append(iso_note)
    if score < 0 and abs(ch) >= 2:
        out["notes"].append(f"The signal moves the opposite way to the one expected ({_f(ch)} %).")
    if math.isfinite(t_half) and t_half > 0.6 * response[1]:
        out["notes"].append(f"Slow change (half reached only at {t_half:.0f} s): a step-like neural response "
                            "is usually faster; check for slow drift.")
    out.update(score=score, change=ch, deepest=deep, deepest_at=deep_at, t_half=t_half, noise=noise, snr=snr,
               change_465=c465, change_405=c405, tier=tier, tier_before_checks=raw_tier)
    return out


def summary_sentence(a):
    if a.get("tier") == "Not testable":
        return a["notes"][0] if a.get("notes") else "Not testable."
    return (f"{a['tier']}: {_f(a['change'])} % ({a['signal'].split(' (')[0]}, {a['response'][0]:+g} to "
            f"{a['response'][1]:+g} s vs {a['baseline'][0]:+g} to {a['baseline'][1]:+g} s); deepest 10 s "
            f"{_f(a['deepest'])} %; 465 {_f(a['change_465'])} %, 405 {_f(a['change_405'])} %; change ÷ noise "
            f"{a['snr']:.1f}" + (f"; half-way at {a['t_half']:.0f} s" if math.isfinite(a.get('t_half', np.nan))
                                  else ""))


def table(assessments):
    """Rows for display, best first. `assessments`: list of (name, animal, assess-dict)."""
    rows = []
    for name, animal, a in assessments:
        rows.append({
            "recording": name, "animal": animal, "grade": a["tier"],
            "change %": a.get("change"), "deepest 10 s %": a.get("deepest"),
            "time to half (s)": a.get("t_half"), "465 %": a.get("change_465"), "405 %": a.get("change_405"),
            "noise %": a.get("noise"), "change ÷ noise": a.get("snr"),
            "event at (s)": a.get("event_s"), "flags": ", ".join(a.get("flags", [])),
            "notes": " ".join(a.get("notes", [])),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {n: i for i, n in enumerate(TIER_NAMES + ["Not testable"])}
    df["_o"] = df["grade"].map(order)
    df["_s"] = [-(a.get("score") if a.get("score") is not None and math.isfinite(a.get("score")) else -1e9)
                for _, _, a in assessments]
    df = df.sort_values(["_o", "_s"]).drop(columns=["_o", "_s"]).reset_index(drop=True)
    return df


def group_summary(assessments):
    """Across recordings (each recording = one animal): mean change, SEM, tests vs 0, grade counts."""
    from scipy import stats
    v = np.array([a["change"] for _, _, a in assessments if a.get("tier") != "Not testable"], float)
    v = v[np.isfinite(v)]
    counts = {n: sum(1 for _, _, a in assessments if a.get("tier") == n) for n in TIER_NAMES + ["Not testable"]}
    out = {"n": int(v.size), "counts": counts}
    if v.size >= 2:
        out.update(mean=float(v.mean()), sem=float(v.std(ddof=1) / np.sqrt(v.size)),
                   median=float(np.median(v)))
        out["t_p"] = float(stats.ttest_1samp(v, 0.0).pvalue) if v.size >= 3 else float("nan")
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(v).pvalue) if v.size >= 5 else float("nan")
        except ValueError:
            out["wilcoxon_p"] = float("nan")
        out["wilcoxon_min_p"] = 2.0 / 2 ** v.size
    return out


def aligned(t, y, event, pre, post, step=0.5):
    """y resampled on a common grid from -pre to +post s around the event (for averaging traces)."""
    tt = np.asarray(t, float) - float(event)
    grid = np.arange(-pre, post + 1e-9, step)
    y = np.asarray(y, float)
    k = np.isfinite(y)
    if k.sum() < 2:
        return grid, np.full(grid.size, np.nan)
    # bin means (robust to 20 Hz noise) rather than point interpolation
    idx = np.floor((tt - (-pre)) / step + 0.5).astype(int)
    ok = k & (idx >= 0) & (idx < grid.size)
    s = np.bincount(idx[ok], y[ok], minlength=grid.size)
    c = np.bincount(idx[ok], minlength=grid.size)
    with np.errstate(invalid="ignore"):
        g = s / c
    g[c == 0] = np.nan
    return grid, g


# =============================================================================
# Sliding-mean guidance (used by every view)
# =============================================================================

_NICE_MIN = [1, 2, 5, 10, 15, 30]


def recommend_smoothing(duration_s):
    """
    -> (switch it on?, window in seconds, one-sentence reason)

    A sliding (running) mean is a low-pass filter: it hides everything faster
    than about its own length. It is the right tool for slow changes (hunger
    state, circadian rhythm, drug effects over hours) in long recordings and
    the wrong tool for anything that happens in seconds (eating, a stimulus,
    a food-presentation drop). Rule used here: about 1/50 of the recording,
    rounded to 1, 2, 5, 10, 15 or 30 min, so the whole recording still has
    ~50 independent points, and never more than 30 min (the window used for
    recordings of days to weeks by Douglass et al. 2024, Cell Metab).
    """
    d = float(duration_s or 0)
    if not math.isfinite(d) or d <= 0:
        return False, 60.0, "Unknown recording length."
    if d < 30 * 60:
        return False, max(5.0, min(30.0, round(d / 60))), (
            f"Off: the recording is {d / 60:.0f} min long, so the changes worth seeing last seconds to minutes and a "
            "sliding mean would blur them. If anything, use a few seconds just to calm noise.")
    w = d / 50.0 / 60.0
    win = min(_NICE_MIN, key=lambda m: abs(math.log(m / w)))
    win = min(win, 30)
    on = d >= 60 * 60
    span = f"{d / 3600:.1f} h" if d >= 3600 else f"{d / 60:.0f} min"
    if not on:
        return False, win * 60.0, (f"Optional: the recording is {span}. A {win}-min sliding mean shows slow drift, "
                                   "but hides events shorter than a few minutes.")
    why = "the Douglass et al. 2024 window for days-long recordings" if win == 30 else "≈ 1/50 of the recording"
    return True, win * 60.0, (f"On, {win} min ({why}): the recording is {span}, long enough that slow trends "
                              "matter. Turn it off to look at single events.")


SMOOTHING_HELP = (
    "A sliding mean averages the signal over a moving window. It shows slow trends (hours-long changes, day/night "
    "rhythm, slow drug effects) and hides everything shorter than the window. Recommended: off for recordings under "
    "30 min and whenever you look at events lasting seconds (eating, food presentation, stimuli); for long "
    "recordings about 1/50 of the length (1 h → 1 min, 4 h → 5 min, 12 h → 15 min, ≥ 24 h → 30 min, as in "
    "Douglass et al. 2024).")
