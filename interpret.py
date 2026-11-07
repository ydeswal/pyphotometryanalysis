#!/usr/bin/env python3
"""
interpret.py
============

Fixed rules for how far a photometry result can be trusted. The rules are
plain code, so the app, the group view and the AI chat all reach the same
verdict, and the AI cannot argue its way to a stronger one.

Why these rules
---------------
* A response is only "significant" against a null: here, the same window
  around random times (permutation test).
* The 405 nm channel is the control for anything that is not calcium:
  movement, chewing, bending of the fibre, blood flow. If it ALSO changes at
  the event, in either direction, the data cannot separate a neural response
  from an artefact:
    - same direction: the classic movement / optical artefact;
    - opposite direction: two possible causes that point different ways.
      GCaMP's 405 nm excitation is a little below its true isosbestic point
      (~410 nm), so a real calcium rise can make 405 dip slightly; but
      haemoglobin absorbs strongly near 405-415 nm, so more blood in the
      tissue (common while eating) also makes 405 fall.
  Either way the 405 change is a reason for caution, never a confirmation.
* The 405 -> 465 fit R^2 says how much of the 465 signal the 405 channel
  could explain. A LOW R^2 means motion correction removed almost nothing;
  it does NOT show there was little motion.
* Few events give unstable means. With animals as the unit (recommended for
  group data, Loewinger et al. 2024 eLife), n is the number of animals.
"""

from __future__ import annotations

import math
import re as _re

LEVELS = ("good", "moderate", "uncertain", "not significant", "not testable")

LEVEL_TEXT = {
    "good": "Reliable: significant, and the 405 control does not change",
    "moderate": "Probably real, with a limitation",
    "uncertain": "Uncertain: significant, but it may not be neural",
    "not significant": "No reliable change",
    "not testable": "Cannot be tested",
}

ALPHA = 0.05
MIN_EVENTS = 8
LOW_R2 = 0.10


def _f(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}g}" if x else "0"


def _p(p):
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(p):
        return "n/a"
    return f"= {p:.2g}" if p >= 0.001 else "< 0.001"


def assess_response(n, effect, p, null_mean=0.0, control_effect=None, control_p=None,
                    fit_r2=None, unit="events", min_n=MIN_EVENTS, versus="random times"):
    """
    Trust level of an event-related change.

      n               number of events (or animals) in the test
      effect          observed mean change (response minus baseline)
      p               p-value of that change (vs random times or vs zero)
      null_mean       mean change around random times (0 for a test vs zero)
      control_effect  405 change vs random times, same units of SD (None = unknown)
      control_p       p-value of the 405 change
      fit_r2          405 -> 465 fit R^2 of the recording (None = unknown)

    Returns {"level", "label", "verdict", "reasons", "must_say"}. `verdict` is
    one sentence to show as is; `reasons` are short facts; `must_say` lists
    the points any written answer has to contain.
    """
    reasons, must = [], []
    try:
        n = int(n or 0)
    except (TypeError, ValueError):
        n = 0
    try:
        sig = p is not None and math.isfinite(float(p)) and float(p) < ALPHA
    except (TypeError, ValueError):
        sig = False
    rel = None
    if effect is not None and null_mean is not None:
        try:
            rel = float(effect) - float(null_mean)
        except (TypeError, ValueError):
            rel = None
    direction = None if rel is None or not math.isfinite(rel) else ("rises" if rel > 0 else "falls")

    try:
        p_ok = p is not None and math.isfinite(float(p))
    except (TypeError, ValueError):
        p_ok = False
    if n < 3 or not p_ok:
        out = {"level": "not testable", "verdict": f"Too few {unit} ({n}) to test.", "reasons": reasons,
               "must_say": ["the test could not be run"]}
        out["label"] = LEVEL_TEXT[out["level"]]
        return out

    level = "good" if sig else "not significant"
    if not sig:
        verdict = (f"No reliable change: the signal {direction or 'changes'} by {_f(rel)} relative to {versus}, "
                   f"p {_p(p)} (n = {n} {unit}).")
        must.append("the change is not statistically reliable")
    else:
        verdict = f"The signal {direction} at the event ({_f(rel)} vs {versus}, p {_p(p)}, n = {n} {unit})."

    # 405 control
    ctrl_sig = None
    if control_effect is not None and control_p is not None:
        try:
            ce, cp = float(control_effect), float(control_p)
        except (TypeError, ValueError):
            ce = cp = float("nan")
        if math.isfinite(ce) and math.isfinite(cp):
            ctrl_sig = cp < ALPHA
            same = rel is not None and math.isfinite(rel) and (ce > 0) == (rel > 0)
            if ctrl_sig:
                bigger = rel is not None and math.isfinite(rel) and abs(ce) > abs(rel)
                if same:
                    reasons.append(f"The 405 control also changes in the SAME direction ({_f(ce)}, p {_p(cp)}): "
                                   "the classic sign of a movement or optical artefact.")
                else:
                    reasons.append(f"The 405 control changes in the OPPOSITE direction ({_f(ce)}, p {_p(cp)}). "
                                   "That has more than one cause (GCaMP's small own response at 405 nm, or more "
                                   "blood in the tissue absorbing 405 nm light), so it neither confirms nor rules "
                                   "out an artefact.")
                if bigger:
                    reasons.append("The 405 change is larger than the 465 response itself.")
                if sig:
                    level = "uncertain"
                must.append("the 405 control also changed significantly, so the response may not be neural")
            else:
                reasons.append(f"The 405 control does not change at the event (p {_p(cp)}).")
    if ctrl_sig is None and sig:
        reasons.append("The 405 control could not be checked.")
        level = "moderate"
        must.append("the 405 control was not checked")

    # motion correction
    if fit_r2 is not None:
        try:
            r2 = float(fit_r2)
        except (TypeError, ValueError):
            r2 = float("nan")
        if math.isfinite(r2) and r2 < LOW_R2:
            reasons.append(f"405 explains only {r2 * 100:.1f}% of the 465 signal (fit R² {r2:.3f}), so motion "
                           "correction removed almost nothing. A low R² does not mean there was little motion.")
            if level == "good":
                level = "moderate"
            must.append("motion correction removed almost nothing (low fit R²), which does not mean low motion")

    if sig and n < min_n:
        reasons.append(f"Only {n} {unit}; the estimate is unstable.")
        if level == "good":
            level = "moderate"

    if level == "uncertain":
        verdict += " Because the 405 control also changed, the data cannot show that this is a neural response."
    elif level == "moderate" and sig:
        verdict += " Probably real, with the limitation below."
    elif level == "good":
        verdict += " The 405 control is flat, so a movement artefact is unlikely."
    return {"level": level, "label": LEVEL_TEXT[level], "verdict": verdict, "reasons": reasons, "must_say": must}


def recording_notes(settings):
    """Plain-language meaning of a recording's quality numbers."""
    s = settings or {}
    out = []
    r2 = s.get("fit_R2")
    if isinstance(r2, (int, float)) and math.isfinite(r2):
        if r2 < LOW_R2:
            out.append(f"fit R² {r2:.3f}: 405 explains almost none of the 465 signal, so motion correction removed "
                       "almost nothing. This does NOT show the recording had little motion; any artefact in 465 "
                       "is still there. Check the 405 control around events.")
        elif r2 < 0.5:
            out.append(f"fit R² {r2:.2f}: 405 explains part of the 465 signal; correction removed some shared changes.")
        else:
            out.append(f"fit R² {r2:.2f}: 405 and 465 share much of their changes; correction matters a lot here.")
    if s.get("fit_slope_clamped"):
        out.append("The 405 fit slope came out negative and was clamped, so the 405 channel was not used to correct.")
    cc = s.get("clock_check") or {}
    if isinstance(cc.get("pc_drift_sec"), (int, float)):
        if s.get("pc_timestamps") or str(s.get("time_axis", "")).lower().startswith("pc"):
            out.append(f"Clock drift {cc['pc_drift_sec']:+.2f} s: the photometry board's own clock against the PC "
                       "clock. The app uses the per-sample PC timestamps, so this is already corrected and does not "
                       "affect alignment with the FED.")
        else:
            out.append(f"Clock drift {cc['pc_drift_sec']:+.2f} s between the photometry board and the PC over the "
                       "recording: event times late in the recording can be off by up to this much.")
    clip = s.get("clipping_fraction")
    if isinstance(clip, (int, float)) and clip > 0.001:
        out.append(f"{clip * 100:.2f}% of samples are clipped (detector saturated); those stretches are unreliable.")
    return out


# Phrases that claim more certainty than photometry data can give.

ALWAYS_BAD = _re.compile(r"\b(prov(?:e|es|ed|en|ing)|definitive(?:ly)?|beyond (?:any )?doubt|undeniabl\w*|"
                         r"irrefutabl\w*|conclusively)\b", _re.I)
OVERCLAIM = _re.compile(
    r"\b(high(?:ly)?[- ]confiden\w*|strong(?:ly)?[- ]confiden\w*|very confident|confirm(?:s|ed|ing)?|"
    r"rul(?:e|es|ed|ing) out|unambiguous(?:ly)?|certainly|genuine(?:ly)?|"
    r"clearly (?:biological|neural|real|a real)|(?:is|are) not (?:an? )?(?:motion |optical )?artifact|"
    r"no (?:motion |optical )?artifact|free of (?:motion )?artifact\w*|"
    r"(?:low|minimal|little|negligible|no) (?:baseline )?motion(?: artifact\w*| noise)?|"
    r"reliabl[ey] (?:neural|biological)|trustworthy)\b", _re.I)


def find_overclaims(text, levels):
    """Phrases in an answer that are stronger than the data checks allow.
    `levels`: trust levels from the data checks used for the answer."""
    hits = [m.group(0) for m in ALWAYS_BAD.finditer(text or "")]
    if any(lv != "good" for lv in levels):
        hits += [m.group(0) for m in OVERCLAIM.finditer(text or "")]
    seen, out = set(), []
    for h in hits:
        k = h.lower()
        if k not in seen:
            seen.add(k)
            out.append(h)
    return out
