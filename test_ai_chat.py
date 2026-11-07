"""Tests for ai_chat: data tools, time parsing, grounding check, and the
question loop against a local stand-in for an OpenAI-compatible API (no
network or API key needed)."""
import json
import tempfile
import threading
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

print("\n=== grounding check ===")
ref = json.dumps({"mean": 0.07907, "max_at": "2026-10-15 11:19:24.3 | rec 00:45:07.9", "p": 0.0005, "n": 12,
                  "ratio": 0.3826})
n, bad = ac.check_grounding("Mean 0.079 %, peak at 11:19:24 (rec 00:45:07), p = 0.0005, 12 events, "
                            "up 38.3% on 2026-10-15.", ref)
check("traceable numbers pass", bad == [], str(bad))
n, bad = ac.check_grounding("The mean was 0.093 and the peak came at 11:21.", ref)
check("invented number and time are flagged", set(bad) == {"0.093", "11:21"}, str(bad))
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

print("\n" + "=" * 60)
print("RESULT: " + ("ALL TESTS PASSED" if not fails else f"{len(fails)} FAILURES: " + ", ".join(fails)))
raise SystemExit(1 if fails else 0)
