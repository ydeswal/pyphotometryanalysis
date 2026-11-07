#!/usr/bin/env python3
"""
ai_chat.py
==========

"Ask the data": a chatbot that answers questions about the recordings and FED3
data loaded in the app, using free AI models.

How it avoids making things up
------------------------------
The AI never sees a picture of a graph and is told never to estimate a
number. It answers by calling the data tools below, which run ordinary numpy /
pandas code on the loaded data (the processed recordings, the raw .ppd files
at full rate, and the FED3 events) and return exact results. Then:

  * arithmetic goes through the `calculate` tool, so derived numbers are
    computed, not guessed;
  * every number and clock time in the answer is checked against the tool
    results; any that cannot be traced is listed under the answer as
    unverified;
  * every tool call and its result can be opened under the answer.

Which free model
----------------
Google Gemini Flash (free tier, about 1,500 requests a day per key, large
token allowance, tool calling) is tried first. If it is out of quota or
unavailable the same question goes to Groq's gpt-oss-120b (free tier, 1,000
requests a day, 8,000 tokens a minute). Any other OpenAI-compatible service
(OpenRouter, a local Ollama) can be added through Streamlit secrets. A typical
question uses 2-3 requests: the model asks for several tools in one round.

Keys come from Streamlit secrets (GEMINI_API_KEY, GROQ_API_KEY, optional
AI_BASE_URL / AI_MODEL / AI_API_KEY, optional CHAT_PASSWORD) or are pasted
into the panel for the current session only.
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import math
import operator
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import fed3
import graph_tools as gt

# =============================================================================
# PROVIDERS
# =============================================================================

PROVIDERS = [
    {
        "id": "gemini", "label": "Google Gemini (free tier)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.8-flash", "model_secret": "GEMINI_MODEL",
        # Each Gemini model has its own quota and its own server load, so when
        # one is busy or used up the same key tries the next.
        "fallback_models": ("gemini-3.6-flash", "gemini-3.5-flash-lite"),
        "fallback_secret": "GEMINI_FALLBACK_MODELS",
        "key_secrets": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "key_url": "https://aistudio.google.com/apikey",
        "extra": {}, "echo_full": True,
        "free_note": "about 1,500 requests/day and 10/minute per key (check your key's limits in AI Studio)",
    },
    {
        "id": "groq", "label": "Groq gpt-oss-120b (free tier)",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b", "model_secret": "GROQ_MODEL",
        "key_secrets": ("GROQ_API_KEY",),
        "key_url": "https://console.groq.com/keys",
        "extra": {"reasoning_effort": "medium"}, "echo_full": False,
        "free_note": "1,000 requests/day, 30/minute, 8,000 tokens/minute",
    },
    {
        "id": "custom", "label": "Custom OpenAI-compatible service",
        "base_url": None, "model": None, "model_secret": "AI_MODEL",
        "key_secrets": ("AI_API_KEY",), "url_secret": "AI_BASE_URL",
        "key_url": None, "extra": {}, "echo_full": False, "free_note": "",
    },
]

MAX_TOOL_ROUNDS = 6
HTTP_TIMEOUT = 90
BUSY_RETRIES = (3, 8)          # seconds to wait before each retry of a busy model


def _secret(name):
    try:
        v = st.secrets.get(name)
    except Exception:
        return None
    return str(v).strip() if v else None


def configured_providers():
    """
    The chain of (service, model) pairs to try, in order: every Gemini model
    on the Gemini key, then Groq, then a custom service. Keys come from
    secrets first, then from keys pasted this session.
    """
    pasted = st.session_state.get("ai_keys", {})
    out = []
    for p in PROVIDERS:
        key = next((k for k in (_secret(s) for s in p["key_secrets"]) if k), None) or pasted.get(p["id"])
        base = p["base_url"] or (_secret(p.get("url_secret", "")) if p.get("url_secret") else None)
        model = _secret(p["model_secret"]) or p["model"]
        if p["id"] == "custom" and not (base and model):
            continue
        if not key and p["id"] != "custom":
            continue
        models = [model]
        fb = _secret(p.get("fallback_secret", "")) if p.get("fallback_secret") else None
        for m in ([x.strip() for x in fb.split(",")] if fb else list(p.get("fallback_models", ()))):
            if m and m not in models:
                models.append(m)
        for m in models:
            out.append(dict(p, key=key or "", base_url=base.rstrip("/"), model=m))
    return out


@st.cache_resource
def _usage_store():
    return {}


def _count_request(pid, model=""):
    store = _usage_store()
    day = _dt.date.today().isoformat()
    k = f"{pid} {model}".strip()
    store.setdefault(day, {}).setdefault(k, 0)
    store[day][k] += 1


def usage_today():
    return dict(_usage_store().get(_dt.date.today().isoformat(), {}))


class ProviderError(Exception):
    def __init__(self, msg, status=None, retry_next=True, retry_after=None, detail="", summary=""):
        super().__init__(msg)
        self.status = status
        self.retry_next = retry_next
        self.retry_after = retry_after
        self.detail = detail          # raw error text from the service
        self.summary = summary        # plain-language summary for the person


def _plain_reason(status, prov):
    return {
        429: "its free-tier limit is used up for now (requests per minute or per day)",
        500: "it had an internal error", 502: "it is temporarily unreachable",
        503: "it is overloaded right now (Google calls this 'high demand')",
        504: "it took too long to answer",
        401: "the API key was rejected (check the key in the secrets)",
        403: "this key may not use this model",
        404: f"the model name '{prov['model']}' is not recognised",
        400: "it refused the request",
    }.get(status, f"it returned error {status}")


def _post(prov, body):
    req = urllib.request.Request(
        prov["base_url"] + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {prov['key']}",
                 "User-Agent": "pyphotometry-app"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode()[:400]
        except Exception:
            pass
        ra = None
        try:
            ra = float(e.headers.get("Retry-After")) if e.headers and e.headers.get("Retry-After") else None
        except (TypeError, ValueError):
            ra = None
        why = _plain_reason(e.code, prov)
        raise ProviderError(f"{prov['model']}: {why}", status=e.code, retry_after=ra, detail=detail,
                            summary=why) from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ProviderError(f"{prov['model']}: could not connect ({e})", detail=str(e),
                            summary="could not connect") from None


def _plain_messages(msgs):
    """Messages stripped to the fields every OpenAI-compatible API accepts."""
    out = []
    for m in msgs:
        if m["role"] == "assistant":
            mm = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                mm["tool_calls"] = [{"id": c["id"], "type": "function",
                                     "function": {"name": c["function"]["name"],
                                                  "arguments": c["function"].get("arguments") or "{}"}}
                                    for c in m["tool_calls"]]
            out.append(mm)
        else:
            out.append({k: m[k] for k in ("role", "content", "tool_call_id") if k in m})
    return out


def _no_answer_message(errors, providers):
    """One readable paragraph for when every model in the chain failed."""
    tried = ", ".join(f"**{m}** ({why})" for m, why in errors)
    busy = all(why.startswith("it is overloaded") or "too long" in why or "unreachable" in why
               for _, why in errors)
    has_groq = any(p["id"] == "groq" for p in providers)
    tip = []
    if busy:
        tip.append("This is on Google's side and usually passes within minutes: press **Ask again**.")
    if any("limit is used up" in why for _, why in errors):
        tip.append("Free-tier limits reset every minute (and daily at midnight Pacific time).")
    if not has_groq:
        tip.append("Adding a free Groq key (`GROQ_API_KEY` in the app's secrets, from "
                   "<https://console.groq.com/keys>) gives a backup that is not affected by Google's load.")
    if any("rejected" in why or "not recognised" in why for _, why in errors):
        tip.append("Check the key and model names in the app's secrets.")
    return f"No AI model could answer. Tried {tried}. " + " ".join(tip)


def complete(providers, msgs, tools, start_at=0, log=None, prepare=None):
    """
    One chat-completion request. A busy model (HTTP 5xx) is retried after a
    short wait; a model that is out of quota, unknown or still busy is skipped
    for the next one in the chain. prepare(i) may return the messages to send
    to chain entry i (used to restart cleanly on a different model).
    Returns (message, chain_index, usage).
    """
    errors = []
    for i in range(start_at, len(providers)):
        prov = providers[i]
        m_i = prepare(i) if prepare else msgs
        body = {"model": prov["model"], "tool_choice": "auto",
                "messages": m_i if prov.get("echo_full") else _plain_messages(m_i)}
        if tools:
            body["tools"] = tools
        else:
            body.pop("tool_choice")
        body.update(prov.get("extra") or {})
        waits = list(BUSY_RETRIES)
        while True:
            try:
                data = _post(prov, body)
                _count_request(prov["id"], prov["model"])
                msg = data["choices"][0]["message"]
                return msg, i, data.get("usage") or {}
            except ProviderError as e:
                busy = e.status is None or 500 <= (e.status or 0) < 600
                if busy and waits:
                    w = min(max(e.retry_after or waits[0], 1.0), 15.0)
                    waits.pop(0)
                    if log:
                        log(f"⏳ {prov['model']} {e.summary}; trying again in {w:.0f} s…")
                    time.sleep(w)
                    continue
                errors.append((prov["model"], e.summary or str(e)))
                if log and i + 1 < len(providers):
                    log(f"↪ {prov['model']} {e.summary}; switching to {providers[i + 1]['model']}…")
                break
            except (KeyError, IndexError, ValueError, TypeError) as e:
                errors.append((prov["model"], f"it sent an unexpected reply ({type(e).__name__})"))
                break
    raise ProviderError(_no_answer_message(errors, providers), retry_next=False)


# =============================================================================
# TIME PARSING
# =============================================================================

class ToolError(Exception):
    pass


_REC_RE = re.compile(r"^\s*(?:rec(?:ording)?(?:\s*time)?\s*|\+)\s*(-?)(\d+):(\d{1,2})(?::(\d{1,2}(?:\.\d*)?))?\s*$", re.I)
_CLOCK_RE = re.compile(r"^\s*(\d{1,2})(?::(\d{2}))?(?::(\d{2}(?:\.\d*)?))?\s*(?:([ap])\.?\s*m\.?)?\s*$", re.I)
_NUMONLY = re.compile(r"^\s*[-+]?\d+(?:\.\d*)?\s*$")


def parse_time(value, clock_start, span, notes=None, is_end=False):
    """
    A time given by the AI -> pandas Timestamp (local wall clock).

      None / "" / "start" / "end"     edge of span
      "1:40 PM" / "13:40" / "1:40:02.5 pm" / "3 pm"
                                      clock time; the date is taken from the
                                      data span (first match if several days)
      "2026-10-15 13:40"              absolute clock time
      "rec 01:25:00" / "+01:25:00"    time into the recording
      1530 / "1530"                   seconds into the recording
    """
    s0, s1 = span
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return s1 if is_end else s0
    if isinstance(value, str) and value.strip().lower() in ("start", "begin"):
        return s0
    if isinstance(value, str) and value.strip().lower() in ("end", "stop"):
        return s1
    if isinstance(value, (int, float)) or (isinstance(value, str) and _NUMONLY.match(value)):
        if clock_start is None:
            raise ToolError("Seconds-into-recording need a recording; give a clock time instead.")
        return clock_start + pd.Timedelta(seconds=float(value))
    v = str(value).strip()
    m = _REC_RE.match(v)
    if m:
        if clock_start is None:
            raise ToolError("Recording time needs a recording; give a clock time instead.")
        sec = int(m.group(2)) * 3600 + int(m.group(3)) * 60 + float(m.group(4) or 0)
        return clock_start + pd.Timedelta(seconds=-sec if m.group(1) else sec)
    m = _CLOCK_RE.match(v)
    if m and (m.group(2) is not None or m.group(4)):
        hour = int(m.group(1))
        ap = (m.group(4) or "").lower()
        if ap:
            if not 1 <= hour <= 12:
                raise ToolError(f"'{v}' is not a valid 12-hour time.")
            hour = (hour % 12) + (12 if ap == "p" else 0)
        tod = pd.Timedelta(hours=hour, minutes=int(m.group(2) or 0), seconds=float(m.group(3) or 0))
        day = s0.normalize() - pd.Timedelta(days=1)
        cands = []
        while day <= s1.normalize() + pd.Timedelta(days=1):
            c = day + tod
            if s0 - pd.Timedelta(hours=1) <= c <= s1 + pd.Timedelta(hours=1):
                cands.append(c)
            day += pd.Timedelta(days=1)
        if not cands:
            raise ToolError(f"{v} is not inside the data ({gt.fmt12(s0, 0, seconds=False)} to "
                            f"{gt.fmt12(s1, 0, seconds=False)}).")
        if len(cands) > 1 and notes is not None:
            notes.append(f"'{v}' occurs on {len(cands)} days; used {cands[0]:%Y-%m-%d}. "
                         "Give the date to choose another day.")
        return cands[0]
    try:
        ts = pd.Timestamp(v)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    except (ValueError, TypeError):
        raise ToolError(f"Could not read the time '{value}'. Use 1:40 PM, 2026-10-15 1:40 PM, "
                        "rec 01:25:00 or seconds.") from None


# =============================================================================
# DATA TOOLS
# =============================================================================

SIGNALS = {
    "dff": ("dF/F (%)", "dF/F (ratio)"),
    "z": ("z-score (dF/F)",),
    "raw465": ("465 raw",),
    "raw405": ("405 raw",),
    "deltaF": ("ΔF (465 − fitted 405)",),
    "filt465": ("465 filtered",),
    "fit405": ("405 fitted to 465",),
    "F0": ("F0 baseline",),
}


def _r(x, sig=4):
    """Round for compact output (significant digits)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return x
    if not math.isfinite(x):
        return None
    if x == 0:
        return 0
    d = sig - int(math.floor(math.log10(abs(x)))) - 1
    v = round(x, max(d, 0))
    return int(v) if float(v).is_integer() and abs(v) < 1e15 else v


def safe_stem(name):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(str(name)).stem)
    return s[:120]


class DataContext:
    """Everything the tools may read, plus what the user is looking at."""

    def __init__(self, results, fed, roi, ppd_paths=None, view=None, fed_map=None,
                 sessions=None, allow_raw=True, pellet_mg=None):
        self.results = list(results or [])
        self.fed = fed if (fed is not None and not fed.events.empty) else None
        self.roi = roi
        self.ppd_paths = dict(ppd_paths or {})
        self.view = dict(view or {})
        self.allow_raw = allow_raw
        self.grams = (pellet_mg or fed3.PELLET_MG_DEFAULT) / 1000.0
        if sessions is None and self.fed is not None and not self.fed.sessions.empty:
            sessions = [s for s in self.fed.sessions["session"] if not str(s).lower().startswith("test")]
        self.sessions = sessions
        self.rec_animal = {}
        for i, r in enumerate(self.results):
            a = (fed_map or {}).get(i)
            if a in (None, "(none)"):
                sid = (r.get("settings") or {}).get("subject_ID", "")
                a = (fed3.guess_animal(r["name"], self.fed.animals, sid) if self.fed is not None else None) \
                    or fed3.animal_from_name(r["name"]) or fed3.animal_from_name(sid)
            self.rec_animal[str(r["name"])] = a
        if self.fed is not None:             # use the FED's spelling (it stores IDs in lower case)
            canon = {x.lower(): x for x in self.fed.animals}
            self.rec_animal = {k: (canon.get(str(a).lower(), a) if a else a) for k, a in self.rec_animal.items()}
        self._cache = {}

    # ---- lookup ---------------------------------------------------------
    def names(self):
        return [str(r["name"]) for r in self.results]

    def rec(self, name):
        if not self.results:
            raise ToolError("No photometry recording is loaded.")
        if name is None and len(self.results) == 1:
            return self.results[0]
        key = str(name or "").strip().lower()
        for r in self.results:
            if str(r["name"]).lower() == key:
                return r
        hits = [r for r in self.results if (self.rec_animal.get(str(r["name"])) or "").lower() == key]
        if len(hits) == 1:
            return hits[0]
        hits = [r for r in self.results if key and key in str(r["name"]).lower()]
        if len(hits) == 1:
            return hits[0]
        raise ToolError(f"Unknown recording '{name}'. Recordings: {self.names()}")

    def animal(self, name):
        if self.fed is None:
            raise ToolError("No FED3 data is loaded.")
        animals = self.fed.animals
        key = str(name or "").strip().lower()
        for a in animals:
            if a.lower() == key:
                return a
        for r, a in self.rec_animal.items():
            if a and key and key in r.lower():
                return a
        if name is None and len(animals) == 1:
            return animals[0]
        raise ToolError(f"Unknown animal '{name}'. FED animals: {animals}")

    def clock_start(self, res):
        k = ("cs", str(res["name"]))
        if k not in self._cache:
            self._cache[k] = gt.clock_start_of(res["df"], res.get("settings"))
        return self._cache[k]

    def recording_for(self, animal):
        for r in self.results:
            if str(self.rec_animal.get(str(r["name"])) or "").lower() == str(animal).lower():
                return r
        return None

    def series(self, res, signal):
        if signal not in SIGNALS:
            raise ToolError(f"Unknown signal '{signal}'. Use one of {list(SIGNALS)}.")
        k = ("s", str(res["name"]), signal)
        if k not in self._cache:
            types = gt.panel_types(res["df"], self.roi)
            kind = next((c for c in SIGNALS[signal] if c in types), None)
            if kind is None:
                raise ToolError(f"Recording '{res['name']}' has no {signal} column. "
                                f"Available: {self.signals_of(res)}")
            t, y, unit, _ = gt._panel_series(res["df"], self.roi, kind, 0.0)
            o = np.argsort(t, kind="stable")
            self._cache[k] = (t[o], y[o], kind, unit)
        return self._cache[k]

    def signals_of(self, res):
        types = gt.panel_types(res["df"], self.roi)
        return [s for s, cands in SIGNALS.items() if any(c in types for c in cands)]

    def rec_span(self, res):
        cs = self.clock_start(res)
        t = pd.to_numeric(res["df"]["time_sec"], errors="coerce")
        return cs + pd.Timedelta(seconds=float(t.min())), cs + pd.Timedelta(seconds=float(t.max()))

    def fed_span(self, animal):
        cov = self.fed.coverage_of(animal, self.sessions)
        ev = self.fed.events[self.fed.events["animal"] == animal]["time"]
        lo = [x for x in (cov["start"].min() if len(cov) else None, ev.min() if len(ev) else None) if x is not None]
        hi = [x for x in (cov["stop"].max() if len(cov) else None, ev.max() if len(ev) else None) if x is not None]
        if not lo:
            raise ToolError(f"No FED data for {animal}.")
        return min(lo), max(hi)

    def window(self, start, end, res=None, animal=None, notes=None):
        if res is not None:
            cs = self.clock_start(res)
            span = self.rec_span(res)
        else:
            cs = None
            rr = self.recording_for(animal) if animal else None
            if rr is not None:
                cs = self.clock_start(rr)
            span = self.fed_span(animal)
        a = parse_time(start, cs, span, notes)
        b = parse_time(end, cs, span, notes, is_end=True)
        if b <= a:
            raise ToolError(f"The window ends ({b}) before it starts ({a}).")
        return a, b, cs

    # ---- formatting -----------------------------------------------------
    @staticmethod
    def tfmt(ts, cs=None):
        if ts is None or pd.isna(ts):
            return None
        s = gt.fmt12(ts, decimals=1)
        if cs is not None:
            s += " | rec " + gt.fmt_hms((pd.Timestamp(ts) - cs).total_seconds(), 1)
        return s

    def sec_fmt(self, sec, cs):
        return self.tfmt(cs + pd.Timedelta(seconds=float(sec)), cs)


# ---- the tool functions -------------------------------------------------

def _sel(t, a_s, b_s):
    return (t >= a_s) & (t <= b_s)


def tool_signal_summary(ctx, recording=None, signal="dff", start=None, end=None):
    res = ctx.rec(recording)
    notes = []
    a, b, cs = ctx.window(start, end, res=res, notes=notes)
    t, y, kind, unit = ctx.series(res, signal)
    a_s, b_s = (a - cs).total_seconds(), (b - cs).total_seconds()
    m = _sel(t, a_s, b_s)
    ok = m & np.isfinite(y)
    out = {"recording": str(res["name"]), "signal": kind, "unit": unit,
           "window": [ctx.tfmt(a, cs), ctx.tfmt(b, cs)], "samples": int(ok.sum()),
           "missing_or_excluded_samples": int(m.sum() - ok.sum())}
    if ok.sum() == 0:
        out["result"] = "no data in this window"
        return out
    tt, yy = t[ok], y[ok]
    imax, imin = int(np.argmax(yy)), int(np.argmin(yy))
    out.update({
        "mean": _r(np.mean(yy)), "sd": _r(np.std(yy, ddof=1)) if yy.size > 1 else None,
        "median": _r(np.median(yy)),
        "min": _r(yy[imin]), "min_at": ctx.sec_fmt(tt[imin], cs),
        "max": _r(yy[imax]), "max_at": ctx.sec_fmt(tt[imax], cs),
        "time_from_min_to_max": (("+" if tt[imax] >= tt[imin] else "-")
                                 + gt.fmt_hms(abs(tt[imax] - tt[imin]), 1)
                                 + f" ({_r(abs(tt[imax] - tt[imin]), 6)} s; max comes "
                                 + ("after" if tt[imax] >= tt[imin] else "before") + " min)"),
        "first_minute_mean": _r(np.mean(yy[tt <= tt[0] + 60])),
        "last_minute_mean": _r(np.mean(yy[tt >= tt[-1] - 60])),
    })
    if yy.size > 10 and tt[-1] > tt[0]:
        out["linear_trend_per_hour"] = _r(np.polyfit((tt - tt[0]) / 3600.0, yy, 1)[0])
    if notes:
        out["notes"] = notes
    return out


def tool_signal_trace(ctx, recording=None, signal="dff", start=None, end=None, points=30):
    res = ctx.rec(recording)
    notes = []
    a, b, cs = ctx.window(start, end, res=res, notes=notes)
    t, y, kind, unit = ctx.series(res, signal)
    pts = int(min(max(int(points or 30), 4), 60))
    a_s, b_s = (a - cs).total_seconds(), (b - cs).total_seconds()
    edges = np.linspace(a_s, b_s, pts + 1)
    idx = np.searchsorted(t, edges)
    rows = []
    for i in range(pts):
        seg = y[idx[i]:idx[i + 1]]
        seg = seg[np.isfinite(seg)]
        rows.append([gt.fmt_clock(cs, edges[i], date=False, decimals=1),
                     _r(seg.mean()) if seg.size else None,
                     _r(seg.min()) if seg.size else None,
                     _r(seg.max()) if seg.size else None])
    out = {"recording": str(res["name"]), "signal": kind, "unit": unit,
           "window": [ctx.tfmt(a, cs), ctx.tfmt(b, cs)], "bin_seconds": _r(edges[1] - edges[0]),
           "columns": ["bin start (clock)", "mean", "min", "max"], "rows": rows}
    if notes:
        out["notes"] = notes
    return out


def tool_find_extremes(ctx, recording=None, signal="dff", start=None, end=None, kind="peaks", n=5,
                       min_separation_s=60):
    res = ctx.rec(recording)
    a, b, cs = ctx.window(start, end, res=res)
    t, y, label, unit = ctx.series(res, signal)
    a_s, b_s = (a - cs).total_seconds(), (b - cs).total_seconds()
    m = _sel(t, a_s, b_s)
    n = int(min(max(int(n or 5), 1), 20))
    df = gt.rank_extremes(t[m], y[m], n=n, min_sep_sec=float(min_separation_s or 60),
                          highest=str(kind).lower().startswith("peak"))
    return {"recording": str(res["name"]), "signal": label, "unit": unit, "kind": kind,
            "window": [ctx.tfmt(a, cs), ctx.tfmt(b, cs)],
            "events": [{"rank": int(r.rank), "time": ctx.sec_fmt(r.time_sec, cs), "value": _r(r.value)}
                       for r in df.itertuples()]}


def tool_compare_windows(ctx, recording=None, signal="dff", windows=None):
    res = ctx.rec(recording)
    t, y, label, unit = ctx.series(res, signal)
    if not windows or len(windows) < 2:
        raise ToolError("Give at least two windows: [{label, start, end}, ...].")
    rows = []
    for w in windows[:8]:
        a, b, cs = ctx.window(w.get("start"), w.get("end"), res=res)
        m = _sel(t, (a - cs).total_seconds(), (b - cs).total_seconds()) & np.isfinite(y)
        yy = y[m]
        rows.append({"label": w.get("label") or f"{gt.fmt12(a, 0, False, False)}-{gt.fmt12(b, 0, False, False)}",
                     "window": [ctx.tfmt(a, cs), ctx.tfmt(b, cs)], "samples": int(yy.size),
                     "mean": _r(yy.mean()) if yy.size else None,
                     "sd": _r(yy.std(ddof=1)) if yy.size > 1 else None,
                     "median": _r(np.median(yy)) if yy.size else None})
    base = rows[0]["mean"]
    for r in rows[1:]:
        if base is not None and r["mean"] is not None:
            r["mean_minus_first"] = _r(r["mean"] - base)
    return {"recording": str(res["name"]), "signal": label, "unit": unit, "windows": rows,
            "note": "Samples within a window are autocorrelated, so no p-value is given; "
                    "use event_aligned_response for a test against random times."}


def _pellets(ctx, animal, cs):
    return fed3.to_seconds(ctx.fed.times(animal, "pellet", ctx.sessions), cs)


def _coverage_sec(ctx, animal, cs):
    cov = ctx.fed.coverage_of(animal, ctx.sessions)
    return list(zip(fed3.to_seconds(cov["start"], cs), fed3.to_seconds(cov["stop"], cs)))


def tool_fed_events(ctx, animal=None, kind="pellet", start=None, end=None, limit=40):
    an = ctx.animal(animal)
    notes = []
    a, b, cs = ctx.window(start, end, animal=an, notes=notes)
    e = ctx.fed.events
    m = (e["animal"] == an) & e["session"].isin(ctx.sessions or e["session"].unique()) \
        & (e["time"] >= a) & (e["time"] <= b)
    if kind and kind != "all":
        m &= e["kind"] == kind
    sub = e.loc[m].sort_values("time")
    lim = int(min(max(int(limit or 40), 1), 100))
    rows = [{"time": ctx.tfmt(r.time, cs), "kind": r.kind, "event": r.event,
             **({"retrieval_s": _r(r.retrieval_s)} if pd.notna(getattr(r, "retrieval_s", np.nan)) else {})}
            for r in sub.head(lim).itertuples()]
    out = {"animal": an, "window": [ctx.tfmt(a, cs), ctx.tfmt(b, cs)],
           "counts": {k: int(v) for k, v in sub["kind"].value_counts().items()},
           "events_shown": len(rows), "events_total": int(len(sub)), "events": rows}
    if kind == "pellet":
        out["food_g"] = _r(len(sub) * ctx.grams)
    if notes:
        out["notes"] = notes
    return out


def tool_meals(ctx, animal=None, start=None, end=None, max_gap_s=60, min_pellets=2, limit=40):
    an = ctx.animal(animal)
    a, b, cs0 = ctx.window(start, end, animal=an)
    ref = cs0 or a.floor("h")
    ps = _pellets(ctx, an, ref)
    ps = ps[(ps >= (a - ref).total_seconds()) & (ps <= (b - ref).total_seconds())]
    meals = fed3.find_meals(ps, float(max_gap_s or 60), int(min_pellets or 2))
    lim = int(min(max(int(limit or 40), 1), 100))
    rows = [{"meal": int(r.meal), "onset": ctx.tfmt(ref + pd.Timedelta(seconds=r.onset), cs0),
             "pellets": int(r.n_pellets), "grams": _r(r.n_pellets * ctx.grams),
             "duration_s": _r(r.duration_s)} for r in meals.head(lim).itertuples()]
    return {"animal": an, "window": [ctx.tfmt(a, cs0), ctx.tfmt(b, cs0)],
            "definition": f"pellets < {max_gap_s} s apart, at least {min_pellets} pellets",
            "meals_total": int(len(meals)), "pellets_in_window": int(ps.size),
            "pellets_in_meals": int(meals["n_pellets"].sum()) if len(meals) else 0,
            "median_meal_pellets": _r(np.median(meals["n_pellets"])) if len(meals) else None,
            "meals_shown": len(rows), "meals": rows}


def tool_intake(ctx, animal=None, start=None, end=None, bin_minutes=60):
    an = ctx.animal(animal)
    a, b, cs0 = ctx.window(start, end, animal=an)
    ref = cs0 or a.floor("h")
    a_s, b_s = (a - ref).total_seconds(), (b - ref).total_seconds()
    bm = float(bin_minutes or 60)
    if (b_s - a_s) / (bm * 60) > 72:
        bm = math.ceil((b_s - a_s) / 72 / 60)
    ps = _pellets(ctx, an, ref)
    cov = _coverage_sec(ctx, an, ref)
    bins = fed3.intake_bins(ps, a_s, b_s, bm * 60, cov)
    logged_h = sum(max(0.0, min(y, b_s) - max(x, a_s)) for x, y in cov) / 3600.0
    n = int(((ps >= a_s) & (ps <= b_s)).sum())
    return {"animal": an, "window": [ctx.tfmt(a, cs0), ctx.tfmt(b, cs0)], "bin_minutes": _r(bm),
            "pellets": n, "food_g": _r(n * ctx.grams), "hours_with_fed_logging": _r(logged_h),
            "pellets_per_logged_hour": _r(n / logged_h) if logged_h else None,
            "columns": ["bin start (clock)", "pellets (null = FED not logging)"],
            "rows": [[(lambda x: f"{x:%m-%d} {gt.fmt12(x, 0, False, False)}")(ref + pd.Timedelta(seconds=r.start)),
                      None if pd.isna(r.pellets) else int(r.pellets)] for r in bins.itertuples()]}


def _window_pair(text, default):
    try:
        a, b = (float(v) for v in str(text or default).split(",")[:2])
        assert a < b
        return a, b
    except Exception:
        raise ToolError(f"Window '{text}' must look like  -100,-50  (start,end seconds).") from None


def tool_event_aligned_response(ctx, animal=None, recording=None, align_to="meals", signal="z",
                                baseline=None, response=None, pre=None, post=None,
                                max_gap_s=60, min_pellets=2):
    an = ctx.animal(animal)
    res = ctx.rec(recording) if recording else ctx.recording_for(an)
    if res is None:
        raise ToolError(f"No photometry recording is matched to {an}.")
    cs = ctx.clock_start(res)
    meal_mode = str(align_to).lower().startswith("meal")
    base = _window_pair(baseline, "-100,-50" if meal_mode else "-40,-20")
    resp = _window_pair(response, "0,50" if meal_mode else "0,20")
    pre = float(pre or (300 if meal_mode else 120))
    post = float(post or (600 if meal_mode else 180))
    t, y, label, unit = ctx.series(res, signal)
    ps = _pellets(ctx, an, cs)
    ps = ps[(ps >= t[0]) & (ps <= t[-1])]
    events = fed3.find_meals(ps, float(max_gap_s), int(min_pellets))["onset"].to_numpy(float) \
        if meal_mode else ps
    if events.size == 0:
        return {"animal": an, "recording": str(res["name"]), "result": "no events inside the recording"}
    mat, tax, kept = fed3.aligned_matrix(t, y, events, pre, post, baseline=base)
    cov = _coverage_sec(ctx, an, cs)
    stats = fed3.permutation_test(t, y, kept, base, resp, allowed_sec=cov, n_perm=2000)
    out = {"animal": an, "recording": str(res["name"]), "signal": label, "unit": unit,
           "aligned_to": "meal onsets" if meal_mode else "every pellet",
           "baseline_window_s": list(base), "response_window_s": list(resp),
           "events_found": int(events.size), "events_with_full_window": int(stats["n"])}
    if stats["n"] >= 3 and np.isfinite(stats["p"]):
        out.update({"mean_response_minus_baseline": _r(stats["observed"]),
                    "same_around_random_times_mean": _r(stats["null_mean"]),
                    "random_times_95pct_range": [_r(stats["ci95"][0]), _r(stats["ci95"][1])],
                    "permutation_p": _r(stats["p"], 3), "random_sets": 2000,
                    "direction": "rises" if stats["observed"] > stats["null_mean"] else "falls"})
        if mat.shape[0]:
            mean_curve = np.nanmean(mat, axis=0)
            k = int(np.nanargmax(np.abs(np.where(tax >= 0, mean_curve, np.nan))))
            out["largest_mean_change_after_event"] = {"seconds_after": _r(tax[k]), "value": _r(mean_curve[k])}
    else:
        out["result"] = "fewer than 3 events with a full window; no test run"
    try:
        import fed_view
        ct, cu = fed_view._control_405(res["df"], ctx.roi)
        if ct is not None and stats["n"] >= 3:
            c = fed3.permutation_test(ct, cu, kept, base, resp, allowed_sec=cov, n_perm=2000)
            if np.isfinite(c["p"]):
                eff = c["observed"] - c["null_mean"]
                same = np.sign(eff) == np.sign(stats["observed"] - stats["null_mean"])
                out["control_405"] = {
                    "change_vs_random_z": _r(eff), "permutation_p": _r(c["p"], 3),
                    "verdict": ("405 also changes in the SAME direction: movement or blood flow may explain part "
                                "of the response" if c["p"] < 0.05 and same else
                                "405 changes in the OPPOSITE direction: interpret with care" if c["p"] < 0.05
                                else "405 does not change at eating")}
    except Exception:
        pass
    deltas = fed3.window_means(t, y, kept, *resp) - fed3.window_means(t, y, kept, *base)
    out["per_event"] = [{"time": ctx.sec_fmt(e, cs), "change": _r(d)} for e, d in zip(kept[:25], deltas[:25])]
    if kept.size > 25:
        out["per_event_note"] = f"first 25 of {kept.size} shown"
    return out


def tool_signal_vs_intake(ctx, animal=None, recording=None, signal="dff", bin_minutes=30, lag_bins=0):
    an = ctx.animal(animal)
    res = ctx.rec(recording) if recording else ctx.recording_for(an)
    if res is None:
        raise ToolError(f"No photometry recording is matched to {an}.")
    cs = ctx.clock_start(res)
    t, y, label, unit = ctx.series(res, signal)
    bs = float(bin_minutes or 30) * 60
    ps = _pellets(ctx, an, cs)
    bins = fed3.intake_bins(ps, float(t[0]), float(t[-1]), bs, _coverage_sec(ctx, an, cs))
    sig = fed3.window_means(t, y, bins["start"].to_numpy(float), 0.0, bs)
    pel = bins["pellets"].to_numpy(float)
    lag = int(lag_bins or 0)
    if lag > 0:
        sig, pel = sig[lag:], pel[:-lag]
    elif lag < 0:
        sig, pel = sig[:lag], pel[-lag:]
    ok = np.isfinite(sig) & np.isfinite(pel)
    out = {"animal": an, "recording": str(res["name"]), "signal": label, "bin_minutes": _r(bs / 60),
           "lag_bins": lag, "bins_used": int(ok.sum())}
    if ok.sum() < 4 or np.std(pel[ok]) == 0:
        out["result"] = "too few bins with FED logging and eating to correlate"
        return out
    from scipy.stats import spearmanr
    out["pearson_r"] = _r(np.corrcoef(sig[ok], pel[ok])[0, 1], 3)
    out["spearman_r"] = _r(spearmanr(sig[ok], pel[ok]).statistic, 3)
    out["note"] = ("Neighbouring bins are not independent, so no p-value is given. lag_bins=1 pairs "
                   "each signal bin with intake in the bin before it.")
    return out


_PPD_CACHE = {}


def _read_ppd_cached(path):
    import photometry_core as pc
    key = (str(path), Path(path).stat().st_mtime)
    if key not in _PPD_CACHE:
        _PPD_CACHE.clear()          # keep at most one file in memory
        _PPD_CACHE[key] = pc.read_ppd(path)
    return _PPD_CACHE[key]


def tool_raw_ppd_window(ctx, recording=None, start=None, end=None, points=30):
    if not ctx.allow_raw:
        raise ToolError("Raw .ppd access is switched off in the chat panel.")
    import photometry_core as pc
    res = ctx.rec(recording)
    path = ctx.ppd_paths.get(str(res["name"]))
    if not path or not Path(path).exists():
        raise ToolError("No .ppd file is available for this recording (it was a CSV, or the upload was "
                        "removed). Use get_signal_summary with signal raw465 / raw405 instead.")
    d = _read_ppd_cached(path)
    clock = d["clock"]
    start_local = clock.get("start_local")
    p0 = pd.Timestamp(start_local) if start_local else ctx.clock_start(res)
    if p0.tzinfo is not None:
        p0 = p0.tz_localize(None)
    fs = float(d["sampling_rate"])
    tax = pc.ppd_time_axis(clock, d["n_samples"])
    span = (p0 + pd.Timedelta(seconds=float(tax[0])), p0 + pd.Timedelta(seconds=float(tax[-1])))
    notes = []
    a = parse_time(start, ctx.clock_start(res), span, notes)
    b = parse_time(end, ctx.clock_start(res), span, notes, is_end=True)
    a_s, b_s = (a - p0).total_seconds(), (b - p0).total_seconds()
    i0, i1 = np.searchsorted(tax, [a_s, b_s])
    if i1 <= i0:
        raise ToolError("No raw samples in this window.")
    cs = ctx.clock_start(res)
    out = {"recording": str(res["name"]), "file": Path(path).name, "sampling_rate_hz": _r(fs),
           "window": [ctx.tfmt(a, cs), ctx.tfmt(b, cs)], "samples": int(i1 - i0),
           "channels": {}}
    pts = int(min(max(int(points or 30), 4), 60))
    edges = np.linspace(i0, i1, pts + 1).astype(int)
    for name, key in (("465 (analog_1, dark-subtracted, V)", "analog_1"),
                      ("405 (analog_2, dark-subtracted, V)", "analog_2")):
        v = np.asarray(d[key][i0:i1], float)
        out["channels"][name] = {"mean": _r(v.mean()), "sd": _r(v.std()), "min": _r(v.min()),
                                 "max": _r(v.max()),
                                 "binned_means": [_r(np.asarray(d[key][edges[j]:edges[j + 1]], float).mean())
                                                  for j in range(pts) if edges[j + 1] > edges[j]]}
        clip = d.get(f"{key}_clipping_fraction")
        if clip is not None:
            out["channels"][name]["clipping_fraction_whole_file"] = _r(clip)
    out["bin_starts_clock"] = [gt.fmt12(p0 + pd.Timedelta(seconds=float(tax[e])), 1, date=False)
                               for e in edges[:-1]]
    for dk in ("digital_1", "digital_2"):
        if dk in d:
            dv = np.asarray(d[dk][i0:i1]).astype(np.int8)
            out[f"{dk}_rising_edges"] = int((np.diff(dv) > 0).sum())
    gaps = [g for g in (clock.get("gaps") or [])]
    if gaps:
        out["lost_sample_gaps_in_file"] = gaps[:5]
    w = clock.get("warnings") or []
    if w:
        out["clock_warnings"] = [str(x)[:200] for x in w[:3]]
    if notes:
        out["notes"] = notes
    return out


def tool_recording_info(ctx, recording=None):
    res = ctx.rec(recording)
    s = res.get("settings") or {}
    keep = ("input_type", "pipeline", "fit_slope_used", "fit_slope_unconstrained", "fit_slope_clamped",
            "fit_R2", "pearson_r_405_vs_465", "F0_median_volts", "lowpass_hz", "median_filter_sec",
            "f0_method", "zscore_mode", "clipping_fraction", "duration_sec", "pc_timestamps",
            "time_axis", "cropped")
    a, b = ctx.rec_span(res)
    cs = ctx.clock_start(res)
    out = {"recording": str(res["name"]), "animal": ctx.rec_animal.get(str(res["name"])),
           "start": ctx.tfmt(a, cs), "end": ctx.tfmt(b, cs),
           "length": gt.fmt_hms((b - a).total_seconds(), 0),
           "signals": ctx.signals_of(res),
           "raw_ppd_available": str(res["name"]) in ctx.ppd_paths,
           "settings": {k: (_r(s[k]) if isinstance(s[k], float) else s[k]) for k in keep if k in s}}
    cc = s.get("clock_check") or {}
    if cc:
        out["clock_check"] = {k: (_r(v) if isinstance(v, float) else v) for k, v in cc.items()
                              if k in ("pc_drift_ppm", "pc_drift_sec", "pc_jitter_ms", "gaps")}
    if s.get("warnings"):
        out["warnings"] = [str(w)[:240] for w in s["warnings"][:6]]
    if s.get("notes"):
        out["notes"] = [str(n)[:200] for n in s["notes"][:4]]
    return out


# ---- safe calculator -----------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"abs": abs, "round": round, "min": min, "max": max, "sqrt": math.sqrt, "log": math.log,
          "log10": math.log10, "exp": math.exp, "mean": lambda *a: sum(a) / len(a)}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        l, r = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(r) > 100:
            raise ToolError("Exponent too large.")
        return _OPS[type(node.op)](l, r)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS \
            and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError("Only numbers, + - * / ** %, and abs round min max sqrt log log10 exp mean are allowed.")


def tool_calculate(ctx, expression=""):
    expr = str(expression).replace("−", "-").replace("×", "*")
    if len(expr) > 300:
        raise ToolError("Expression too long.")
    try:
        val = _eval(ast.parse(expr, mode="eval"))
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"Could not calculate: {e}") from None
    return {"expression": expr, "result": _r(val, 6)}


def tool_time_difference(ctx, start=None, end=None, recording=None):
    """Time from start to end. Both are read like any other time argument."""
    res = None
    if recording or ctx.results:
        try:
            res = ctx.rec(recording)
        except ToolError:
            res = None
    if res is not None:
        cs = ctx.clock_start(res)
        span = ctx.rec_span(res)
    elif ctx.fed is not None and ctx.fed.animals:
        cs = None
        spans = [ctx.fed_span(a) for a in ctx.fed.animals]
        span = (min(x[0] for x in spans), max(x[1] for x in spans))
    else:
        raise ToolError("No data loaded to anchor the times to.")
    if start is None or end is None:
        raise ToolError("Give both start and end.")
    a = parse_time(start, cs, span)
    b = parse_time(end, cs, span, is_end=True)
    d = (b - a).total_seconds()
    return {"start": ctx.tfmt(a, cs), "end": ctx.tfmt(b, cs), "difference": ("-" if d < 0 else "")
            + gt.fmt_hms(abs(d), 1), "seconds": _r(d, 7), "minutes": _r(d / 60, 6), "hours": _r(d / 3600, 6)}


# ---- tool registry -------------------------------------------------------

_T = {"type": "string", "description": "1:40 PM | 2026-10-15 1:40 PM | rec 01:25:00 | seconds; omit = all"}
_SIG = {"type": "string", "enum": list(SIGNALS)}
_REC = {"type": "string", "description": "recording name or animal ID"}
_AN = {"type": "string", "description": "animal ID"}


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


TOOLS = [
    _fn("get_signal_summary", "Exact statistics of one signal over a time window: mean, sd, median, min/max "
        "with their times, first/last-minute means, linear trend per hour.",
        {"recording": _REC, "signal": _SIG, "start": _T, "end": _T}, ["signal"]),
    _fn("get_signal_trace", "The shape of a signal over a window as binned mean/min/max (4-60 bins). "
        "Use to describe how it changes over time.",
        {"recording": _REC, "signal": _SIG, "start": _T, "end": _T,
         "points": {"type": "integer", "description": "number of bins, 4-60 (default 30)"}}, ["signal"]),
    _fn("find_extremes", "Highest peaks or lowest troughs of a signal as distinct events, with times.",
        {"recording": _REC, "signal": _SIG, "start": _T, "end": _T,
         "kind": {"type": "string", "enum": ["peaks", "troughs"]},
         "n": {"type": "integer", "description": "how many, 1-20"},
         "min_separation_s": {"type": "number", "description": "minimum seconds between events (default 60)"}},
        ["signal"]),
    _fn("compare_windows", "Mean, sd and median of a signal in 2-8 time windows, and each window's mean minus "
        "the first window's mean.",
        {"recording": _REC, "signal": _SIG, "windows": {"type": "array", "items": {
            "type": "object", "properties": {"label": {"type": "string"}, "start": _T, "end": _T}}}},
        ["signal", "windows"]),
    _fn("get_fed_events", "FED3 events of one animal in a window: pellets taken, pokes, dispenses; with times "
        "and counts.",
        {"animal": _AN, "kind": {"type": "string", "enum": ["pellet", "poke", "dispensed", "all"]},
         "start": _T, "end": _T, "limit": {"type": "integer", "description": "max rows, default 40"}},
        ["animal"]),
    _fn("get_meals", "Meals of one animal (pellets grouped by the gap between them) with onset times, pellets, "
        "grams and duration.",
        {"animal": _AN, "start": _T, "end": _T,
         "max_gap_s": {"type": "number", "description": "default 60"},
         "min_pellets": {"type": "integer", "description": "default 2"},
         "limit": {"type": "integer"}}, ["animal"]),
    _fn("get_intake", "Food intake of one animal: total pellets, grams, hours of FED logging, and pellets per "
        "time bin (null where the FED was not logging).",
        {"animal": _AN, "start": _T, "end": _T, "bin_minutes": {"type": "number", "description": "default 60"}},
        ["animal"]),
    _fn("event_aligned_response", "Does the photometry signal change around meal onsets or pellets more than "
        "around random times? Returns mean response-minus-baseline, the same for random times, a permutation "
        "p-value, a 405-channel control and per-event changes.",
        {"animal": _AN, "recording": _REC, "align_to": {"type": "string", "enum": ["meals", "pellets"]},
         "signal": _SIG, "baseline": {"type": "string", "description": "seconds 'start,end', default -100,-50 "
                                                                       "for meals, -40,-20 for pellets"},
         "response": {"type": "string", "description": "seconds 'start,end', default 0,50 / 0,20"},
         "pre": {"type": "number"}, "post": {"type": "number"},
         "max_gap_s": {"type": "number"}, "min_pellets": {"type": "integer"}}, ["animal"]),
    _fn("signal_vs_intake", "Correlation between the binned photometry signal and pellets eaten per bin "
        "(only bins with FED logging).",
        {"animal": _AN, "recording": _REC, "signal": _SIG, "bin_minutes": {"type": "number"},
         "lag_bins": {"type": "integer", "description": "1 = intake in the bin before the signal"}},
        ["animal"]),
    _fn("get_raw_ppd_window", "The original .ppd file at full sampling rate in a window: 465 and 405 "
        "dark-subtracted volts (mean, sd, min, max, binned means), digital input edges, lost-sample gaps and "
        "clock warnings. Use for questions about the raw recording, clipping, signal quality.",
        {"recording": _REC, "start": _T, "end": _T, "points": {"type": "integer"}}, []),
    _fn("get_recording_info", "Processing settings and quality checks of a recording: pipeline, 405 fit slope "
        "and R², F0, clipping, clock drift and gaps, warnings.", {"recording": _REC}, []),
    _fn("time_difference", "How long from one time to another (clock times, dates or recording times). "
        "Use for any time arithmetic.", {"start": _T, "end": _T, "recording": _REC}, ["start", "end"]),
    _fn("calculate", "Arithmetic on numbers from tool results (differences, ratios, percentages, unit "
        "conversions). Always use this instead of mental arithmetic.",
        {"expression": {"type": "string", "description": "e.g. (0.412-0.298)/0.298*100"}}, ["expression"]),
]

TOOL_FUNCS = {
    "get_signal_summary": tool_signal_summary, "get_signal_trace": tool_signal_trace,
    "find_extremes": tool_find_extremes, "compare_windows": tool_compare_windows,
    "get_fed_events": tool_fed_events, "get_meals": tool_meals, "get_intake": tool_intake,
    "event_aligned_response": tool_event_aligned_response, "signal_vs_intake": tool_signal_vs_intake,
    "get_raw_ppd_window": tool_raw_ppd_window, "get_recording_info": tool_recording_info,
    "calculate": tool_calculate, "time_difference": tool_time_difference,
}


def available_tools(ctx):
    skip = set()
    if not ctx.results:
        skip |= {"get_signal_summary", "get_signal_trace", "find_extremes", "compare_windows",
                 "event_aligned_response", "signal_vs_intake", "get_raw_ppd_window", "get_recording_info"}
    if ctx.fed is None:
        skip |= {"get_fed_events", "get_meals", "get_intake", "event_aligned_response", "signal_vs_intake"}
    if not ctx.allow_raw or not ctx.ppd_paths:
        skip.add("get_raw_ppd_window")
    return [t for t in TOOLS if t["function"]["name"] not in skip]


def run_tool(ctx, name, args):
    fn = TOOL_FUNCS.get(name)
    if fn is None:
        return {"error": f"Unknown tool {name}."}
    try:
        return fn(ctx, **{k: v for k, v in (args or {}).items() if v is not None})
    except ToolError as e:
        return {"error": str(e)}
    except TypeError as e:
        return {"error": f"Bad arguments for {name}: {e}"}
    except Exception as e:                       # never crash the chat on a tool bug
        return {"error": f"{name} failed: {type(e).__name__}: {e}"}


# =============================================================================
# CONTEXT FOR THE MODEL
# =============================================================================

def catalogue(ctx, focus=None):
    """Compact description of the loaded data and of what is on screen."""
    L = [f"Today: {gt.fmt12(pd.Timestamp.now(), 0, seconds=False)}. ROI: {ctx.roi}."]
    if ctx.results:
        L.append("PHOTOMETRY RECORDINGS:")
        for r in ctx.results:
            try:
                a, b = ctx.rec_span(r)
                cs = ctx.clock_start(r)
                s = r.get("settings") or {}
                L.append(f"- {r['name']}: animal {ctx.rec_animal.get(str(r['name'])) or '?'}; "
                         f"{ctx.tfmt(a, cs)} to {ctx.tfmt(b, cs)} (length {gt.fmt_hms((b - a).total_seconds(), 0)}); "
                         f"signals {', '.join(ctx.signals_of(r))}; raw .ppd "
                         f"{'yes' if str(r['name']) in ctx.ppd_paths and ctx.allow_raw else 'no'}"
                         + (f"; 405 fit R² {_r(s['fit_R2'], 3)}" if isinstance(s.get("fit_R2"), (int, float)) else "")
                         + ("; 405 slope was negative and clamped" if s.get("fit_slope_clamped") else "")
                         + (f"; {len(s['warnings'])} processing warning(s)" if s.get("warnings") else ""))
            except Exception as e:
                L.append(f"- {r['name']}: (could not summarise: {e})")
    else:
        L.append("No photometry recording loaded.")
    if ctx.fed is not None:
        L.append(f"FED3 DATA: animals {', '.join(ctx.fed.animals)}; sessions used: "
                 f"{', '.join(map(str, ctx.sessions or []))}; pellet mass {_r(ctx.grams * 1000)} mg.")
        for an in ctx.fed.animals:
            cov = ctx.fed.coverage_of(an, ctx.sessions)
            n = len(ctx.fed.times(an, "pellet", ctx.sessions))
            segs = "; ".join(f"{r.start:%m-%d} {gt.fmt12(r.start, 0, False, False)} to "
                             f"{r.stop:%m-%d} {gt.fmt12(r.stop, 0, False, False)}" for r in cov.itertuples())
            L.append(f"- {an}: {n} pellets in used sessions; FED logging {segs or 'unknown'}")
    else:
        L.append("No FED3 data loaded.")
    v = ctx.view
    if v:
        L.append("ON SCREEN NOW: " + "; ".join(f"{k} = {val}" for k, val in v.items() if val not in (None, "", [])))
    if focus:
        L.append("THE USER IS ASKING ABOUT: " + ", ".join(focus))
    return "\n".join(L)


SYSTEM_RULES = """You are the data assistant inside a fibre-photometry (pyPhotometry, 465 nm GCaMP with 405 nm \
isosbestic control) and FED3 feeding-device analysis web app used by neuroscientists.

Accuracy rules (most important):
1. Every number, time, count or p-value you write must come from a tool result in this conversation or from the \
DATA CATALOGUE. Never estimate, recall or invent values, and never read values off a graph you imagine.
2. Never do arithmetic yourself. For any difference, ratio, percentage or unit conversion call `calculate`; \
for any time span call `time_difference` (get_signal_summary already gives the time from min to max).
3. If the tools cannot answer, say so plainly and say what data or setting would be needed.
4. Write clock times in 12-hour form with AM/PM (e.g. 3:57 PM), as the tools give them, and add the time \
into the recording (rec HH:MM:SS, a duration) when relevant. Never write clock times in 24-hour form.
5. Separate what the data show from interpretation. FED gaps mean "not logged", not "not eating". Mention the \
405 control when you discuss a response. Never claim causation from correlation.

Signals: dff = dF/F, z = z-scored dF/F, raw465 / raw405 = raw channels at the app's export rate (the original \
.ppd at full rate is get_raw_ppd_window), deltaF = 465 - fitted 405, filt465, fit405, F0 = baseline.

Efficiency:
- Plan first and ask for all independent tool calls in one turn.
- Use defaults unless the user asks otherwise; do not repeat a call you already made.
- Questions about "this graph" or "here" refer to ON SCREEN NOW and THE USER IS ASKING ABOUT.

Answer format: lead with the direct answer, then the key numbers, then at most two caveats. Plain language, \
under 200 words unless asked for more. Markdown allowed, no tables wider than 4 columns."""


def system_prompt(ctx, focus=None):
    return SYSTEM_RULES + "\n\nDATA CATALOGUE\n" + catalogue(ctx, focus)


# =============================================================================
# GROUNDING CHECK
# =============================================================================

_TIME_TOK = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?![\d:])(?:\s*[AaPp]\.?\s?[Mm]\b\.?)?")
_HOUR_TOK = re.compile(r"(?<![\d:.])(\d{1,2})\s*([AaPp])\.?\s?[Mm]\b\.?")


def _time_forms(tok):
    """Normalised forms of a time token: as written (zero-padded) and, when it
    has AM/PM, converted to 24 h. Each at minute, second and full precision."""
    m = re.match(r"(\d{1,2}):(\d{2})(?::(\d{2})(\.\d+)?)?\s*(?:([AaPp])\.?\s?[Mm]\.?)?", tok.strip())
    if not m:
        return set()
    h, mi, se, fr, ap = int(m.group(1)), m.group(2), m.group(3), m.group(4) or "", (m.group(5) or "").lower()
    hours = {h}
    if ap and 1 <= h <= 12:
        hours = {(h % 12) + (12 if ap == "p" else 0)}
    out = set()
    for hh in hours:
        base = f"{hh:02d}:{mi}"
        out.add(base)
        if se is not None:
            out.add(f"{base}:{se}")
            if fr:
                out.add(f"{base}:{se}{fr}")
    return out
_DATE_TOK = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUM_TOK = re.compile(r"(?<![\w.])[-−+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")


def _nums(text):
    out = []
    for m in _NUM_TOK.finditer(_HOUR_TOK.sub(" ", _TIME_TOK.sub(" ", _DATE_TOK.sub(" ", text)))):
        s = m.group(0).replace("−", "-").replace(",", "")
        try:
            out.append(float(s))
        except ValueError:
            pass
    return out


def _decimals(tok):
    tok = tok.replace(",", "")
    return len(tok.split(".")[1]) if "." in tok and "e" not in tok.lower() else 0


def check_grounding(answer, reference_text):
    """
    Which numbers and clock times in the answer cannot be found in the tool
    results / catalogue / question? A number matches when a reference value
    rounds to it at the answer's precision (also as a %, or without sign).
    List numbering and the integers 0-2 are not checked. Returns
    (n_checked, unverified_tokens).
    """
    ref_nums = np.array(_nums(reference_text) or [np.nan])
    ref_times, ref_bare = set(), set()     # 24-hour forms; 12-hour forms written without AM/PM
    for m in _TIME_TOK.finditer(reference_text):
        tok = m.group(0)
        ref_times |= _time_forms(tok)
        bare = re.sub(r"\s*[AaPp]\.?\s?[Mm]\.?$", "", tok)      # '3:57 PM' may be written '3:57'
        if bare != tok:
            ref_bare |= _time_forms(bare)
    for m in _HOUR_TOK.finditer(reference_text):          # '6 PM' in the question
        ref_times |= _time_forms(f"{m.group(1)}:00 {m.group(2)}M")
    ref_hours = {t[:2] for t in ref_times}
    ref_dates = set(_DATE_TOK.findall(reference_text))
    checked, bad = 0, []
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", answer)          # list numbering
    for m in _DATE_TOK.finditer(text):
        checked += 1
        if m.group(0) not in ref_dates:
            bad.append(m.group(0))
    no_dates = _DATE_TOK.sub(" ", text)
    for m in _TIME_TOK.finditer(no_dates):
        tok = m.group(0).strip().rstrip(".")
        forms = _time_forms(tok)
        full = max(forms, key=len) if forms else tok
        has_ap = bool(re.search(r"[AaPp]\.?\s?[Mm]$", tok))
        checked += 1
        if not (full in ref_times or (not has_ap and full in ref_bare)):
            bad.append(tok)
    for m in _HOUR_TOK.finditer(_TIME_TOK.sub(" ", no_dates)):
        f = _time_forms(f"{m.group(1)}:00 {m.group(2)}M")
        checked += 1
        if not f or next(iter(f))[:2] not in ref_hours:
            bad.append(m.group(0).strip().rstrip("."))
    for m in _NUM_TOK.finditer(_HOUR_TOK.sub(" ", _TIME_TOK.sub(" ", no_dates))):
        tok = m.group(0)
        try:
            x = float(tok.replace("−", "-").replace(",", ""))
        except ValueError:
            continue
        if float(x).is_integer() and 0 <= x <= 2:
            continue
        checked += 1
        tol = 0.5 * 10 ** (-_decimals(tok)) * 1.0001 + 1e-12
        ok = False
        for cand in (ref_nums, np.abs(ref_nums), ref_nums * 100.0, np.abs(ref_nums) * 100.0):
            if np.any(np.abs(cand - abs(x)) <= tol) or np.any(np.abs(cand - x) <= tol):
                ok = True
                break
        if not ok:
            bad.append(tok)
    return checked, bad


# =============================================================================
# ONE QUESTION
# =============================================================================

def ask(question, history, ctx, providers, focus=None, log=None):
    """
    Answer one question. history: list of {"q", "a", "ref"} from earlier
    turns (only their text is resent, not their tool results). Returns a dict:
    answer, trace, provider, requests, tokens, checked, unverified, error.
    """
    sys_msg = system_prompt(ctx, focus)
    msgs = [{"role": "system", "content": sys_msg}]
    for h in history[-6:]:
        msgs += [{"role": "user", "content": h["q"]}, {"role": "assistant", "content": h["a"]}]
    msgs.append({"role": "user", "content": question})
    base_msgs = list(msgs)
    tools = available_tools(ctx)
    trace, ref_parts = [], [sys_msg, question] + [h.get("ref", "") + "\n" + h["q"] for h in history[-6:]]
    n_req, tokens, pidx = 0, 0, 0
    calls_seen = {}
    answer = None
    owner = [None]        # chain index whose tool-call turns are in msgs

    def prepare(i):
        """Messages for chain entry i. A different model than the one that made
        the tool calls so far starts over, with the results already computed
        handed to it as data, so no model sees another model's internal fields."""
        if owner[0] is None or owner[0] == i:
            return msgs
        done = [{"tool": t["tool"], "args": t["args"], "result": t["result"]} for t in trace]
        return base_msgs[:-1] + [{"role": "user", "content": (
            question + "\n\n(Already computed by the data tools for this question; use these results and "
                       "call more tools only if needed:)\n" + json.dumps(done, default=str)[:12000])}]

    try:
        for rnd in range(MAX_TOOL_ROUNDS + 1):
            use_tools = tools if rnd < MAX_TOOL_ROUNDS else None
            if rnd == MAX_TOOL_ROUNDS:
                msgs.append({"role": "user", "content": "Tool budget used up. Answer now from the tool results "
                                                        "above, and say what is still unknown."})
            msg, new_idx, usage = complete(providers, msgs, use_tools, start_at=pidx, log=log, prepare=prepare)
            if owner[0] is not None and new_idx != owner[0]:
                msgs = prepare(new_idx)               # continue from the restarted conversation
            pidx = new_idx
            if msg.get("tool_calls"):
                owner[0] = pidx
            n_req += 1
            tokens += int(usage.get("total_tokens") or 0)
            msg = {k: v for k, v in msg.items() if v is not None}
            msg.setdefault("content", "")
            msgs.append(msg if providers[pidx].get("echo_full") else _plain_messages([msg])[0])
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                break
            for c in calls:
                name = c["function"]["name"]
                raw = c["function"].get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                except json.JSONDecodeError:
                    args = {}
                sig = name + json.dumps(args, sort_keys=True)
                if sig in calls_seen:
                    out = calls_seen[sig]
                else:
                    if log:
                        log(f"🔧 {name}({', '.join(f'{k}={v}' for k, v in args.items())})")
                    out = run_tool(ctx, name, args)
                    calls_seen[sig] = out
                text = json.dumps(out, separators=(",", ":"), default=str)
                trace.append({"tool": name, "args": args, "result": out})
                ref_parts += [text, json.dumps(args, default=str)]
                msgs.append({"role": "tool", "tool_call_id": c.get("id") or name, "content": text})
    except ProviderError as e:
        return {"answer": None, "error": str(e), "trace": trace, "requests": n_req, "tokens": tokens,
                "provider": None, "checked": 0, "unverified": [], "ref": "", "retry": True}
    ref = "\n".join(ref_parts)
    checked, bad = check_grounding(answer or "", ref)
    return {"answer": answer or "(The model gave no answer.)", "error": None, "trace": trace,
            "provider": f"{providers[pidx]['label']} · {providers[pidx]['model']}", "requests": n_req,
            "tokens": tokens, "checked": checked, "unverified": bad,
            "ref": "\n".join(json.dumps(t["result"], default=str) for t in trace)[:20000]}


# =============================================================================
# STREAMLIT PANEL
# =============================================================================

def _suggestions(ctx, focus):
    fx = [str(f).removeprefix("FED ") for f in (focus or [])]
    an = next((a for a in (ctx.rec_animal.get(f) or f for f in fx)
               if ctx.fed is not None and a in ctx.fed.animals),
              ctx.fed.animals[0] if ctx.fed is not None and ctx.fed.animals else None)
    rec = next((f for f in (focus or []) if f in ctx.names()), ctx.names()[0] if ctx.names() else None)
    out = []
    if an and ctx.results:
        out.append(f"Does {an}'s signal rise after meals, and is the 405 control clean?")
    if rec:
        out.append(f"Describe the {rec} recording: biggest peaks, trend, and any quality problems.")
    if an:
        out.append(f"When did {an} eat the most, and how big were its meals?")
    if ctx.fed is not None and len(ctx.fed.animals) > 1:
        out.append("Compare food intake and the meal response across all animals.")
    return out[:4]


def render_panel(results, fed, roi, ppd_paths=None, view=None, fed_map=None, sessions=None,
                 pellet_mg=None):
    """The chat panel. Call once per page run; it reruns on its own (fragment)."""
    with st.expander("💬 Ask the data (AI)", expanded=bool(st.session_state.get("ai_history"))):
        _panel(results, fed, roi, ppd_paths, view, fed_map, sessions, pellet_mg)


@st.fragment
def _panel(results, fed, roi, ppd_paths, view, fed_map, sessions, pellet_mg):
    pw = _secret("CHAT_PASSWORD")
    if pw and st.session_state.get("ai_unlocked") != pw:
        got = st.text_input("Chat password", type="password", key="ai_pw")
        if got == pw:
            st.session_state["ai_unlocked"] = pw
            st.rerun(scope="fragment")
        st.caption("The lab has set a password for the AI chat.")
        return

    provs = configured_providers()
    with st.popover("⚙ AI settings"):
        st.markdown("**Free AI keys** (used only in this browser session unless set in the app's secrets)")
        keys = st.session_state.setdefault("ai_keys", {})
        for p in PROVIDERS[:2]:
            in_secrets = any(_secret(s) for s in p["key_secrets"])
            if in_secrets:
                st.caption(f"{p['label']}: key set by the app ✓")
                continue
            v = st.text_input(f"{p['label']} key", value=keys.get(p["id"], ""), type="password",
                              key=f"ai_key_{p['id']}", help=f"Free key: {p['key_url']}  ·  {p['free_note']}")
            keys[p["id"]] = v.strip()
        chain = configured_providers()
        if chain:
            st.caption("Tried in this order: " + " → ".join(p["model"] for p in chain))
        st.toggle("Let the AI read the raw .ppd files (full rate)", value=True, key="ai_allow_raw")
        use = usage_today()
        if use:
            st.caption("Requests today on this server: " + ", ".join(f"{k} {v}" for k, v in use.items()))
        st.caption("On free tiers, Google may use what is sent to improve its products and reviewers may "
                   "read it; Groq's terms differ. Only computed summaries of your data are sent, never "
                   "whole files, but do not use the free tier for data you must keep confidential.")
    provs = configured_providers()
    if not provs:
        st.info("Add a free key in **⚙ AI settings** to start: a Google Gemini key from "
                "<https://aistudio.google.com/apikey> (recommended), and optionally a Groq key from "
                "<https://console.groq.com/keys> as a fallback.")
        return
    if not results and (fed is None or fed.events.empty):
        st.info("Run an analysis or load FED3 data first; the AI answers from the loaded data.")
        return

    ctx = DataContext(results, fed, roi, ppd_paths, view, fed_map, sessions,
                      allow_raw=st.session_state.get("ai_allow_raw", True), pellet_mg=pellet_mg)
    options = ctx.names() + ([f"FED {a}" for a in ctx.fed.animals] if ctx.fed is not None else [])
    auto = [o for o in options if o in (view or {}).get("focus_default", [])]
    ss = st.session_state
    # Follow what is on screen until the user picks something themselves.
    if "ai_focus" not in ss or (ss.get("ai_focus") == ss.get("ai_focus_auto") and ss.get("ai_focus_auto") != auto):
        ss["ai_focus"] = auto
    ss["ai_focus"] = [o for o in ss["ai_focus"] if o in options]
    ss["ai_focus_auto"] = auto
    focus = st.multiselect("Graphs / data to ask about (one or more)", options, key="ai_focus",
                           help="Follows the graph on screen. Pick several to compare; leave empty to let "
                                "the AI use everything loaded.")
    hist = st.session_state.setdefault("ai_history", [])

    pending = None
    retry = st.session_state.pop("ai_retry", None)
    if retry is not None and 0 <= retry < len(hist):
        pending = hist.pop(retry)["q"]          # replace the failed answer

    for i, h in enumerate(hist):
        with st.chat_message("user"):
            st.markdown(h["q"])
        with st.chat_message("assistant"):
            _show_answer(h, i if h.get("error") else None)

    if not hist:
        cols = st.columns(2)
        for i, s in enumerate(_suggestions(ctx, focus)):
            if cols[i % 2].button(s, key=f"ai_sugg_{i}", width="stretch"):
                pending = s
    q = st.chat_input("Ask about these graphs, the raw recording or the FED data…", key="ai_input")
    q = q or pending
    bar = st.columns([1, 5])
    if hist and bar[0].button("Clear chat", key="ai_clear"):
        st.session_state["ai_history"] = []
        st.rerun(scope="fragment")
    if not q:
        return
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        with st.status("Working it out from the data…", expanded=True) as status:
            res = ask(q, [h for h in hist if not h.get("error")], ctx, provs, focus=focus, log=st.write)
            status.update(label="Done" if not res["error"] else "Could not answer",
                          state="complete" if not res["error"] else "error", expanded=False)
    hist.append({"q": q, **res, "a": res["answer"] or ""})
    st.rerun(scope="fragment")


def _show_answer(h, idx=None):
    if h.get("error"):
        st.warning(h["error"])
        if idx is not None and st.button("↻ Ask again", key=f"ai_retry_{idx}"):
            st.session_state["ai_retry"] = idx
            st.rerun(scope="fragment")
        return
    st.markdown(h["a"])
    if h.get("unverified"):
        st.markdown(f'<div class="pp-note warn">⚠ Not found in any computed result, so treat as unverified: '
                    f'{", ".join(h["unverified"][:12])}</div>', unsafe_allow_html=True)
    elif h.get("checked"):
        st.caption(f"✓ All {h['checked']} numbers and times in this answer were traced to computed results.")
    with st.expander(f"How this was answered · {len(h.get('trace', []))} data calls · "
                     f"{h.get('requests', 0)} AI requests · {h.get('provider') or ''}"):
        for t in h.get("trace", []):
            st.markdown(f"**{t['tool']}** `{json.dumps(t['args'], default=str)}`")
            st.json(t["result"], expanded=False)
        if not h.get("trace"):
            st.caption("Answered from the data catalogue without extra calculations.")
