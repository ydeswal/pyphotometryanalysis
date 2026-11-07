"""Tests for ai_chat: data tools, time parsing, grounding check, and the
question loop against a local stand-in for an OpenAI-compatible API (no
network or API key needed)."""
import json
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

import ai_chat as ac
import fed3

fails = []


def check(name, cond, extra=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))
    if not cond:
        fails.append(name)


# ---- synthetic recording (2 h, 10 Hz) with a response after each meal ----
rng = np.random.default_rng(0)
c0 = pd.Timestamp("2026-10-15 10:34:16.400")
t = np.arange(0, 7200, 0.1)
meals = np.array([600.0, 2400.0, 4200.0, 6000.0])
pel = np.sort(np.concatenate([m + np.array([0, 20, 45]) for m in meals]))
resp = sum(3.0 * np.exp(-np.clip(t - m - 20, 0, None) / 120) * (t > m) for m in meals)
dff = resp + 0.3 * rng.standard_normal(t.size)
df = pd.DataFrame({"time_sec": t, "BLA_dFF_percent": dff, "BLA_z_dFF": (dff - dff.mean()) / dff.std(),
                   "BLA_sig_raw": 1.0 + dff / 100, "BLA_uv_raw": 0.5 + 0.002 * rng.standard_normal(t.size)})
df["clock_time"] = (c0 + pd.to_timedelta(df["time_sec"], unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3]

HDR = ("MM:DD:YYYY hh:mm:ss,Library_Version,Session_type,Device_Number,Battery_Voltage,Motor_Turns,FR,"
       "Event,Active_Poke,Left_Poke_Count,Right_Poke_Count,Pellet_Count,Block_Pellet_Count,Retrieval_Time,"
       "InterPelletInterval,Poke_Time,PC_Timestamp,PC_Received,PC_Received_Unix")
tmp = Path(tempfile.mkdtemp())
sess = tmp / "FED3_data" / "refeed_20261015_103500"
sess.mkdir(parents=True)
rows = [HDR]
for i, x in enumerate(pel):
    ts = c0 + pd.Timedelta(seconds=float(x))
    rows.append(f"{ts:%m/%d/%Y %H:%M:%S},1.17.3,PC_Free,1,4.2,1,0,Pellet,Left,0,0,{i + 1},0,2.1,nan,nan,"
                f"{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{ts:%Y-%m-%d %H:%M:%S.%f}"[:-3] + f",{ts.timestamp():.3f}")
(sess / "FED001_m10_data.csv").write_text("\n".join(rows) + "\n")
(sess / "session_info.json").write_text(json.dumps({
    "session_name": "refeed", "start_pc_time": "2026-10-15 10:35:00.000", "stop_pc_time": "2026-10-15 12:34:00.000",
    "utc_offset_hours": -5.0, "feds": {"COM5": {"fed_number": 1, "animal_id": "m10", "info": {}, "last_status": {}}}}))
fd = fed3.load_fed([sess / "FED001_m10_data.csv", sess / "session_info.json"], tmp / "work")
NAME = "m10_fasted-2026-10-15-103416"
ctx = ac.DataContext([{"name": NAME, "df": df, "settings": {"fit_R2": 0.9}}], fd, "BLA")

print("=== time parsing ===")
span = (c0, c0 + pd.Timedelta(hours=2))
check("clock time", ac.parse_time("11:00", c0, span) == pd.Timestamp("2026-10-15 11:00"))
check("recording time", ac.parse_time("rec 00:10:00", c0, span) == c0 + pd.Timedelta(minutes=10))
check("seconds", ac.parse_time(60, c0, span) == c0 + pd.Timedelta(seconds=60))
check("open end = end of data", ac.parse_time(None, c0, span, is_end=True) == span[1])
try:
    ac.parse_time("23:00", c0, span)
    check("time outside the data is refused", False)
except ac.ToolError:
    check("time outside the data is refused", True)
check("12-hour clock time", ac.parse_time("11:05 am", c0, span) == pd.Timestamp("2026-10-15 11:05")
      and ac.parse_time("12:10 PM", c0, span) == pd.Timestamp("2026-10-15 12:10")
      and ac.parse_time("12 pm", c0, span) == pd.Timestamp("2026-10-15 12:00"))
check("date with 12-hour time", ac.parse_time("2026-10-15 11:05 AM", c0, span) == pd.Timestamp("2026-10-15 11:05"))
span2 = (c0, c0 + pd.Timedelta(hours=30))
notes = []
ac.parse_time("12:00", c0, span2, notes)
check("ambiguous clock time on a multi-day recording is noted", len(notes) == 1, str(notes))

print("\n=== data tools ===")
o = ac.run_tool(ctx, "get_signal_summary", {"recording": "m10", "signal": "dff", "start": "11:00", "end": "11:30"})
a = (pd.Timestamp("2026-10-15 11:00") - c0).total_seconds()
b = (pd.Timestamp("2026-10-15 11:30") - c0).total_seconds()
m = (t >= a) & (t <= b)
check("summary mean is exact", abs(o["mean"] - dff[m].mean()) < 1e-3 * max(1, abs(dff[m].mean())), str(o.get("mean")))
check("summary max time given in both clocks", "| rec " in o["max_at"])
o = ac.run_tool(ctx, "get_meals", {"animal": "m10"})
check("meals found", o["meals_total"] == 4, str(o.get("meals_total")))
check("first meal onset", o["meals"][0]["onset"].startswith("2026-10-15 10:44:16.4"), o["meals"][0]["onset"])
o = ac.run_tool(ctx, "event_aligned_response", {"animal": "m10", "signal": "dff"})
check("meal response detected", o.get("direction") == "rises" and o["events_with_full_window"] == 4, str(o)[:200])
o = ac.run_tool(ctx, "get_intake", {"animal": "m10", "bin_minutes": 30})
check("intake total", o["pellets"] == pel.size and abs(o["food_g"] - pel.size * 0.02) < 1e-9)
o = ac.run_tool(ctx, "calculate", {"expression": "(4.2-3.1)/3.1*100"})
check("calculator", abs(o["result"] - 35.4839) < 1e-3, str(o))
o = ac.run_tool(ctx, "calculate", {"expression": "__import__('os').getcwd()"})
check("calculator refuses code", "error" in o)
o = ac.run_tool(ctx, "get_signal_summary", {"recording": "m99", "signal": "dff"})
check("unknown recording gives a usable error", "Recordings:" in o.get("error", ""))
o = ac.run_tool(ctx, "get_raw_ppd_window", {"recording": "m10"})
check("raw .ppd tool explains when there is no .ppd", "No .ppd" in o.get("error", ""))
names = [x["function"]["name"] for x in ac.available_tools(ctx)]
check("raw .ppd tool not offered without a .ppd", "get_raw_ppd_window" not in names)

ctx_case = ac.DataContext([{"name": NAME, "df": df, "settings": {}}], fd, "BLA", fed_map={0: "M10"})
check("animal IDs match regardless of case", ctx_case.recording_for("m10") is not None
      and ctx_case.rec_animal[NAME] == "m10")

o = ac.run_tool(ctx, "get_meals", {"animal": "m10"})
check("tool output times are 12-hour", o["meals"][0]["onset"] == "2026-10-15 10:44:16.4 AM | rec 00:10:00.0",
      o["meals"][0]["onset"])

print("\n=== grounding check ===")
ref = json.dumps({"mean": 0.07907, "max_at": "2026-10-15 11:19:24.3 | rec 00:45:07.9", "p": 0.0005, "n": 12,
                  "ratio": 0.3826})
n, bad = ac.check_grounding("Mean 0.079 %, peak at 11:19:24 (rec 00:45:07), p = 0.0005, 12 events, "
                            "up 38.3% on 2026-10-15.", ref)
check("traceable numbers pass", bad == [], str(bad))
n, bad = ac.check_grounding("The mean was 0.093 and the peak came at 11:21.", ref)
check("invented number and time are flagged", set(bad) == {"0.093", "11:21"}, str(bad))
ref12 = json.dumps({"onset": "2026-11-05 12:18:06.0 AM | rec 08:20:34.4", "max_at": "2026-11-04 1:40:02.6 PM"})
n, bad = ac.check_grounding("It began at 12:18 AM (rec 08:20:34); the peak was at 1:40:02 PM, i.e. 13:40, or 1:40.",
                            ref12)
check("12-hour, 24-hour and bare forms of the same time all pass", bad == [] and n == 5, f"{n} {bad}")
n, bad = ac.check_grounding("It began at 12:18 PM and peaked at 3 PM.", ref12)
check("wrong AM/PM and invented hour are flagged", set(bad) == {"12:18 PM", "3 PM"}, str(bad))
n, bad = ac.check_grounding("1. First point\n2. Second point with 1 or 2 items", ref)
check("list numbering and 0-2 are not checked", bad == [], str(bad))

print("\n=== question loop against a stand-in API ===")
seen = []


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append((self.server.server_port, body))
        if self.server.server_port == PORTS[0]:
            self.send_response(429)
            self.end_headers()
            self.wfile.write(b'{"error":"quota"}')
            return
        msgs = body["messages"]
        if msgs[-1]["role"] == "user":
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "extra_content": {"google": {"thought_signature": "S"}},
                 "function": {"name": "get_meals", "arguments": json.dumps({"animal": "m10"})}},
                {"id": "c2", "type": "function",
                 "function": {"name": "get_meals", "arguments": json.dumps({"animal": "m10"})}}]}
        else:
            meals = json.loads([m for m in msgs if m["role"] == "tool"][0]["content"])
            msg = {"role": "assistant",
                   "content": f"m10 ate {meals['meals_total']} meals, first at {meals['meals'][0]['onset'][11:19]}; "
                              f"about 9.9 g."}
        out = json.dumps({"choices": [{"message": msg}], "usage": {"total_tokens": 100}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


servers = [HTTPServer(("127.0.0.1", 0), H) for _ in range(2)]
PORTS = [s.server_port for s in servers]
for s in servers:
    threading.Thread(target=s.serve_forever, daemon=True).start()
provs = [dict(id="gemini", label="A", base_url=f"http://127.0.0.1:{PORTS[0]}", model="a", key="k", extra={},
              echo_full=True, model_secret="X"),
         dict(id="groq", label="B", base_url=f"http://127.0.0.1:{PORTS[1]}", model="b", key="k",
              extra={"reasoning_effort": "medium"}, echo_full=False, model_secret="Y")]
r = ac.ask("How many meals did m10 eat?", [], ctx, provs)
check("falls back when the first service is out of quota", r["provider"].startswith("B"), str(r["provider"]))
check("answer arrives in 2 AI requests", r["requests"] == 2, str(r["requests"]))
check("identical tool call run once", sum(1 for x in r["trace"]) == 2 and r["trace"][0]["result"] is r["trace"][1]["result"])
check("answer numbers checked; invented 9.9 flagged", r["unverified"] == ["9.9"], str(r["unverified"]))
last = [b for p, b in seen if p == PORTS[1]][-1]
check("fallback service gets plain messages", "extra_content" not in json.dumps(last["messages"]))
check("provider extras sent", last.get("reasoning_effort") == "medium")

# Echo of provider-specific fields (e.g. Gemini thought signatures) when echo_full is on.
seen.clear()
provs[1]["echo_full"] = True
r = ac.ask("How many meals?", [], ctx, provs[1:])
last = [b for p, b in seen][-1]
check("echo_full keeps provider fields on the assistant turn", "thought_signature" in json.dumps(last["messages"]))
for s in servers:
    s.shutdown()

print("\n=== busy models, retries and switching models mid-question ===")
ac.BUSY_RETRIES = (0.01,)
calls = []


class B(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        model = body["model"]
        calls.append((model, body))
        n = sum(1 for m, _ in calls if m == model)
        msgs = body["messages"]
        busy = {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}
        if model == "busy" or (model == "flaky" and n == 1) or (model == "dies" and n >= 2):
            return self._send(503, busy)
        last = msgs[-1]
        if last["role"] == "user" and "Already computed" not in last["content"]:
            return self._send(200, {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "t1", "type": "function", "extra_content": {"google": {"thought_signature": "SIG-" + model}},
                 "function": {"name": "get_meals", "arguments": json.dumps({"animal": "m10"})}}]}}]})
        if last["role"] == "tool":
            meals = json.loads(last["content"])
        else:
            meals = json.loads(last["content"].split(":)\n", 1)[1])[0]["result"]
        return self._send(200, {"choices": [{"message": {"role": "assistant",
                                                          "content": f"{meals['meals_total']} meals."}}]})


srv = HTTPServer(("127.0.0.1", 0), B)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{srv.server_port}"
mk = lambda m: dict(id="gemini", label="G", base_url=url, model=m, key="k", extra={}, echo_full=True,
                    model_secret="X")
ac._health_store().clear()
calls.clear()
r = ac.ask("meals?", [], ctx, [mk("flaky")])
check("a busy model is retried and then answers", r["answer"] == "4 meals." and
      [m for m, _ in calls].count("flaky") == 3, str([m for m, _ in calls]))
ac._health_store().clear()
calls.clear()
r = ac.ask("meals?", [], ctx, [mk("busy"), mk("ok")])
check("a model that stays busy is skipped for the next", r["answer"] == "4 meals." and r["provider"].endswith("ok"))
ac._health_store().clear()
calls.clear()
r = ac.ask("meals?", [], ctx, [mk("dies"), mk("ok")])
sent_to_ok = [b for m, b in calls if m == "ok"]
check("switch mid-question restarts on the new model with the computed results",
      r["answer"] == "4 meals." and len(sent_to_ok) == 1
      and "Already computed" in sent_to_ok[0]["messages"][-1]["content"], str(r.get("answer")))
check("the new model never sees the old model's internal fields",
      "SIG-dies" not in json.dumps(sent_to_ok[0]["messages"]))
check("the restarted answer is still checked against the earlier results", r["unverified"] == [], str(r["unverified"]))
ac._health_store().clear()
calls.clear()
t0 = time.perf_counter()
r = ac.ask("meals?", [], ctx, [mk("busy"), mk("ok")])
r = ac.ask("meals again?", [], ctx, [mk("busy"), mk("ok")])
check("a model that just failed is not tried first next time",
      [m for m, _ in calls].count("busy") == 1 and r["answer"] == "4 meals.", str([m for m, _ in calls]))
check("switching to the next model does not wait", time.perf_counter() - t0 < 5, f"{time.perf_counter() - t0:.1f} s")
chain = ac.order_chain([mk("busy"), mk("ok"), mk("fresh")])
check("chain order: last good first, failing last", [p["model"] for p in chain] == ["ok", "fresh", "busy"],
      str([p["model"] for p in chain]))
ac._health_store().clear()
calls.clear()
r = ac.ask("meals?", [], ctx, [mk("busy")])
check("all busy: plain-language message with a tip", r["error"] and "overloaded" in r["error"]
      and "Ask again" in r["error"] and "Groq" in r["error"], str(r["error"]))
srv.shutdown()

print("\n=== interpretation checks ===")
ac._health_store().clear()
OVER = "High confidence: the opposite 405 shift confirms the signal is genuine, and the low R² shows low motion."
GOOD = ("The signal rises after meals, but the 405 control also changes, so this may not be neural. "
        "Motion correction removed almost nothing.")


class O(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        calls.append(("o", body))
        if msgs[-1]["role"] == "user" and "not consistent with the app's fixed data checks" in msgs[-1]["content"]:
            msg = {"role": "assistant", "content": GOOD}
        elif msgs[-1]["role"] == "user":
            msg = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "e1", "type": "function", "function": {"name": "event_aligned_response",
                                                              "arguments": json.dumps({"animal": "m10"})}}]}
        else:
            msg = {"role": "assistant", "content": OVER}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)


# a recording whose 405 channel jumps at every meal
df405 = df.copy()
uv = 0.5 + 0.002 * rng.standard_normal(t.size)
for m_ in meals:
    uv -= 0.02 * np.exp(-np.clip(t - m_ - 20, 0, None) / 120) * (t > m_)
df405["BLA_uv_raw"] = uv
ctx405 = ac.DataContext([{"name": NAME, "df": df405, "settings": {"fit_R2": 0.009}}], fd, "BLA")
osrv = HTTPServer(("127.0.0.1", 0), O)
threading.Thread(target=osrv.serve_forever, daemon=True).start()
calls.clear()
r = ac.ask("Does m10 respond to eating?", [], ctx405,
           [dict(id="gemini", label="G", base_url=f"http://127.0.0.1:{osrv.server_port}", model="o", key="k",
                 extra={}, echo_full=True, model_secret="X")])
lv = [c["level"] for c in r["checks"]]
check("tool gives a fixed verdict: uncertain when 405 moves", "uncertain" in lv, str(lv))
check("overclaiming draft is sent back and corrected", r["corrected"] and r["answer"] == GOOD and not r["overclaims"],
      str(r.get("overclaims")))
check("the correction request lists the bad phrases",
      "High confidence" in calls[-1][1]["messages"][-1]["content"])
osrv.shutdown()
check("signed numbers must match the sign", ac.check_grounding("It changed by +0.30.", json.dumps({"x": -0.30}))[1]
      == ["+0.30"] and ac.check_grounding("It fell by 0.30.", json.dumps({"x": -0.30}))[1] == [])

print("\n=== group tools ===")
df2 = df.copy()
df2["BLA_dFF_percent"] = df["BLA_dFF_percent"] * 0.7 + 0.3 * rng.standard_normal(t.size)
df2["BLA_z_dFF"] = (df2["BLA_dFF_percent"] - df2["BLA_dFF_percent"].mean()) / df2["BLA_dFF_percent"].std()
ctxg = ac.DataContext([{"name": NAME, "df": df, "settings": {"fit_R2": 0.4}},
                       {"name": "m10_day2", "df": df2, "settings": {"fit_R2": 0.4}}], fd, "BLA",
                      groups={NAME: "Day 1", "m10_day2": "Day 2"})
o = ac.run_tool(ctxg, "group_meal_response", {"signal": "z"})
check("group tool runs per group", set(o.get("groups", {})) == {"Day 1", "Day 2"}, str(o)[:300])
check("no between-group test with one animal per group", not o.get("comparisons"), str(o.get("comparisons")))
check("one animal: group verdict says it cannot be tested",
      all(g["data_check"]["level"] == "not testable" for g in o["groups"].values()))
o = ac.run_tool(ctxg, "meal_patterns", {})
check("meal pattern tool", len(o["animals"]) == 2 and o["animals"][0]["meals"] == 4, str(o["animals"])[:200])

print("\n=== time arithmetic ===")
o = ac.run_tool(ctx, "get_signal_summary", {"recording": "m10", "signal": "dff"})
t_min = pd.Timestamp(o["min_at"][:21]); t_max = pd.Timestamp(o["max_at"][:21])
check("summary gives time from min to max", "time_from_min_to_max" in o and
      abs(float(o["time_from_min_to_max"].split("(")[1].split(" s")[0]) - abs((t_max - t_min).total_seconds())) < 0.2,
      o.get("time_from_min_to_max"))
o = ac.run_tool(ctx, "time_difference", {"start": "10:44:16", "end": "rec 01:00:00"})
check("time_difference mixes clock and recording time", o["difference"] == "00:50:00.4", str(o))

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
