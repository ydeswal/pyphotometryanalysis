# pyPhotometry Analysis (Sweeney Lab)

Streamlit app for analysing pyPhotometry `.ppd` and `.csv` fiber photometry
recordings: motion correction, dF/F0, z-scoring, long-timescale visualisation,
and lickometer analysis.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Every formula is explained in [`METHODS_EXPLAINED.md`](METHODS_EXPLAINED.md)**,
which is also shown inside the app in the **Methods & math** tab.

---

## Run it on your own computer

Download or clone this repository, then double-click:

| | |
|---|---|
| **macOS** | `run_local.command` (first time: right-click → **Open** → **Open**, because macOS blocks downloaded scripts) |
| **Windows** | `run_local.bat` |
| **Linux** | `bash run_local.sh` |

The first run creates a private Python environment in `.venv` and installs the
packages (1–3 minutes). After that it starts in seconds. The app opens at
<http://localhost:8501>. Keep the window open while you work, and close it to
stop. You need Python 3.11 or 3.12 from <https://www.python.org/downloads/>.
On a local machine there is no cloud memory cap, and uploads up to 4 GB are
allowed.

---

## 16 October 2026: FED3 feeding data on the photometry clock

Sidebar → **FED3 feeding data**: upload the zipped `FED3_data` folder from the
FED3 controller (or its `*_data.csv`, `*_pc_events.csv`, `session_info.json`
files). Then **Display → Feeding (FED3)**. Recordings are matched to animals by
the ID in their names (`m10_fasted_feds-…` → `m10`). Change the match in
*Recording ↔ animal* if a name doesn't carry the ID.

| View | What it shows |
|---|---|
| **Photometry above, food below** | dF/F or z on top; pellets per bin (bars) and cumulative food in grams underneath, on one clock axis |
| **One graph: signal + eating** | every pellet as a dot ON the trace at the moment it was taken, plus eating rate (pellets/h) on a second axis |
| **Aligned to eating** | the signal around meal onsets or every pellet: mean ± SEM, every event as a heatmap, a test against random times, a 405 control, and a per-event table (CSV) |
| **All animals** | cumulative food of every animal together, and a summary table |

Every graph has the 📏 ruler: it counts the **pellets and grams eaten between
its two points**. The ▲▼ Peaks tool also works here.

**How the data are read** (`fed3.py`):

- **Event times** use each FED row's `PC_Timestamp`. That is the PC
  wall-clock time computed on the FED at the moment of the event, the same
  clock as the photometry `clock_time`.
- **Intake** is counted from Pellet *events*, not `Pellet_Count`, which
  restarts with every SD file and every session.
- **Animals** are identified by the ID in the file name, not the FED device
  number (three FEDs all report "FED000").
- **Periods with no FED logging** (e.g. 13:40–19:36 on 15 Oct) are shaded
  grey and treated as *unknown*, not as zero intake. Random comparison times
  are only drawn from logged periods.
- **Test sessions:** sessions whose name starts with `test` are left out by
  default.
- **Two FEDs with the same animal ID:** this is flagged. In `longi`, COM10 and
  COM16 were both labelled m16; they wrote into the same files and one
  `pc_events` line was cut off. COM10 logged no events, so m16's data are
  intact. Fix it at source by giving every FED a distinct ID.

**Aligned to eating, the numbers:**

- Response − baseline is computed per event. The defaults follow Douglass et
  al. 2025: −100…−50 s vs 0…50 s for meals, and −40…−20 s vs 0…20 s for single
  pellets.
- That mean is compared with the same statistic at 2,000 sets of random times
  when the FED was logging, away from real events, giving a permutation p.
- **405 control:** the same test on the 405 channel (drift removed, z-scored).
  - If 405 moves the **same** way at eating, movement, chewing or blood flow
    may explain the response.
  - If it moves the **opposite** way (m11), interpret with care.
- A **meal** is pellets less than 60 s apart, at least 2 pellets (both
  adjustable).

`python test_fed3.py` covers reading (counter resets, duplicate rows,
damaged lines, zip safety, coverage) and the analysis (bins, meals, rates,
aligned matrices, permutation test on signals with and without a response).

---

## 15 October 2026 (b): side-by-side graphs, peaks, time ruler, memory

**Why the hosted app crashed.** Streamlit Community Cloud gives an app at
most 2.7 GB of memory, shared by everyone using it at that moment. The
previous version:

- built **all five graphs** for every recording on every click (`st.tabs`
  renders every tab), with every sample of every trace;
- kept every uploaded file in server memory;
- built the full CSV and ZIP downloads on every click, even if nobody
  pressed the download button;
- converted the channels to 64-bit copies during analysis, and held the
  whole raw file in RAM.

| One 24 h recording (m10, 20 Hz) | Before | After |
|---|---|---|
| Data sent to the browser | 81 MB | **4.5 MB** |
| Time to first graph | 78 s | **17 s** |
| Server memory afterwards | 516 MB | **~400 MB** |
| Analysis peak, 48 h at 130 Hz (360 MB file) | 1,526 MB | **1,005 MB** |

What changed: graphs are thinned before they are drawn, which keeps each
stretch's first, last, highest and lowest point, so every visible peak and
trough survives. Only the view you pick is built. Uploads go straight to disk.
Downloads are built only when clicked. The reader memory-maps the file, and the
analysis stays in 32-bit floats (results identical to within 2×10⁻⁷).
*Graph display → Graph detail* sets how many points are drawn.
Calculations, peak values and downloads always use every sample.

**Display: two options at the top.**

1. **One recording at a time.** The same views as before (Overview, Raw
   channels, Main dFF, Z-score sliding mean, Z-score range analysis, …),
   chosen with a row of buttons. Only the chosen view is built.
2. **Side-by-side graphs.** 2–6 graphs, 1–3 per row. Each graph has its own
   recording and graph type: dF/F, z-score, z running mean, ΔF, raw 465/405,
   filtered 465, fitted 405 or F0. Options:
   - **Elapsed** or **Clock time** axis; clock time lines recordings up by
     time of day.
   - **Same time window in every graph**, a slider in hours or in clock time.
   - **Same y-scale for the same graph type.**

**▲▼ Peaks** (dF/F and z-score graphs, every side-by-side graph, the lick
graph):
- Marks the current **ceiling** (highest point) and **floor** (lowest point).
- Lists the top 5/10/20 peaks and troughs as distinct events, with elapsed
  and clock time. *Min. gap between events* stops one broad transient from
  filling the list.
- Tick **Remove** on any row to blank that event. It is blanked from where
  the trace leaves its median to where it returns, capped at ± N s. The
  ceiling or floor then moves to the next one, and the y-axis can refit to
  what is left.
- **Fix y floor** and **Fix y ceiling** work independently.
- **Restore all removed** undoes every removal.
- All of this is display-only: it never changes data, z-scores or downloads.

**📏 Ruler** (every graph):
- Turn it on, then click two points on the graph. With the default hover you
  don't need to hit the line exactly; the click takes the time under the
  cursor.
- Shows both times (elapsed and wall clock), the time between them (h:m:s,
  s, min, h), and the mean, min and max of the signal in between.
- A third click starts a new measurement.
- Timing resolution is the spacing of the drawn points; narrow the visual
  crop window, or raise *Graph detail*, for finer timing.

Both tools run inside `st.fragment`, so clicking a graph reruns only that
graph, not the whole app. They use Streamlit's own Plotly click events and
need no extra package. (`streamlit-plotly-events`, which another proposal
used, has not been updated since 2021.)

`python test_graph_tools.py` tests thinning, peak ranking, removal and time
formatting.

---

## 15 October 2026: PC-timestamped `.ppd` files

**Symptom.** A recording made with the modified acquisition program that
writes PC timestamps showed **twice its real duration** (47:51:36 for a
23:55:48 file), a sawtooth instead of a trace, a negative 405 slope with
R² = 0.000, and about 2% "clipping".

**Cause.** That program stores each sample as the usual 4 data words
(8 bytes) **followed by an 8-byte PC timestamp**, 16 bytes in all; the
header says so (`"pc_timestamps": true, "bytes_per_sample": 16`). Every
standard reader assumes 8 bytes, including this app's previous version and the
official pyPhotometry importer. It took each timestamp for two more samples of
light. Nothing in the file was damaged.

**Fix.** `photometry_core.read_ppd` reads the layout from the header, and
detects it from the data if an unlabelled file turns up. It also recovers
files whose last sample was cut off mid-write. On `m10_fasted-2026-10-14`:

| | Before | After |
|---|---|---|
| Duration | 47:51:36 | **23:55:48** |
| 405→465 slope | −0.738, clamped to 0 | **+2.66** |
| Fit R² | 0.000 | **0.892** |
| Clipping | 2.09% | **0%** |
| Start time | none | **2026-10-14 10:23:00.423** (UTC−5) |

**Wall-clock time is kept.** Every processed CSV now has `clock_unix` and
`clock_time` columns. Hovering any trace shows its clock time, and a line under
the summary cards gives the start and end. Files without PC timestamps get
`clock_time` from the header `date_time`. A lick CSV with clock timestamps
(e.g. LIQ HD `pc_clock`) is now **shifted onto the photometry clock**. Before,
it was laid over the photometry as if both had started at the same instant.

**Which clock?** Each PC timestamp is the moment the PC *received* a sample
(jitter about 3 ms). The stamps come from the Windows performance counter,
anchored to the wall clock once at the start. On m10 they gained **2.38 s over
the day** relative to the board's sample counter. pyPhotometry writes
`end_time` from the Windows wall clock when the file closes. That time is
**0.30 s after** the last sample on the sample clock, but **2.08 s before** the
last PC stamp, which is impossible unless the stamps drifted. So by default:

- **Time axis = board sample clock, anchored to the PC clock at the first
  sample.** This matches the wall clock that FED3 `PC_Timestamp` and LIQ HD
  `pc_clock` use.
- The PC stamps are used for the start time, to **detect lost samples** (a
  forward jump in PC time that the sample counter does not show; the time axis
  then includes the gap), and for the cross-check above. All of it is reported
  under *Processing notes*.
- *Processing method → Time axis* switches to following the stamps instead.

**Recovering files for other software.** `python3 convert_ppd.py file.ppd`
writes `file_standard.ppd` (a standard 8-byte file that any reader opens
correctly, checked against the official pyPhotometry importer), plus
`file_pc_timestamps.csv.gz` (every raw stamp) and `file_clock_report.json`. The
standard file keeps the PC start time in its header, so this app still shows
clock times for it. The original file is never modified.
`python3 check_ppd.py file.ppd` reports the layout, the clock checks and a
verdict for any file.

Also fixed: a file truncated mid-frame used to "fall back" to the other frame
layout whenever that divided the data, which silently misread it. The declared
layout is now kept and only the incomplete frame is dropped.

`test_core.py` sections 19–23 cover all of this, including the real m10 file
when it is in the folder.

---

## October 2026 update: calculations now follow Douglass et al. 2025

Default pipeline = STAR Methods of Douglass et al. (2025) *Cell Metabolism*
37:708-722, with unstated details taken from pyPhotometry (Akam & Walton 2019),
which the paper cites.

| Fix | Before | After |
|---|---|---|
| Negative 405 slope | Accepted. On M16 the slope was -17.8, which subtracted the real signal and amplified 405 noise | Constrained to >= 0 with a warning (toggle in sidebar) |
| F0 edge handling | 0.001 Hz filter held the first sample (F0 = 1.47 V on M16) | Mirror padding; F0 = 1.79 V (correct) |
| F0 decimation | Point sampling (aliasing) | Block averaging |
| 405 fit | IRLS default | OLS ("linear best fit", as in the paper); IRLS optional |
| Median / filter order | 1 s / order 3 | 5 samples / order 2 (pyPhotometry) |
| dF/F label | Ratio shown as "%" | True percent |
| Time axis | Hours only | s / min / h automatically |
| Export rate | Always 1 Hz | Auto: full rate for short files |
| Diagnostics | None | Slope, R^2, F0, warnings above every recording |

New sidebar section **Processing method (formulas)** exposes every setting,
plus an alternative pyPhotometry pipeline (double-exponential bleaching
correction). `.ppd` runs from the app skip the unused matplotlib PNGs.

`python test_core.py` runs 18 test groups, including OLS = scipy linregress,
the negative-slope guard, F0 edges, filter equivalence, and a check on the M16
recording if it is present.

---

## Earlier changes (kept for reference)

## File layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `ana.py` | Batch `.ppd` command-line pipeline (called by `app.py`) |
| `photometry_core.py` | **All maths.** Single source of truth, including `.ppd` reading and the clock model |
| `convert_ppd.py` | PC-timestamped `.ppd` → standard `.ppd` + timestamps (for other software) |
| `check_ppd.py` | Diagnose any `.ppd`: layout, clock, duration, verdict |
| `graph_tools.py` | Side-by-side graphs, peaks/troughs, time ruler, graph thinning |
| `fed3.py` | Reads FED3 controller output; intake, meals, event alignment, permutation test |
| `fed_view.py` | The "Feeding (FED3)" view |
| `run_local.command` / `.bat` / `.sh` | Start the app on your own computer |
| `lickometer.py` | Lick ingestion, bout detection, peri-event alignment |
| `theme.py` | Styling and Plotly defaults |
| `.streamlit/config.toml` | Base theme |
| `test_core.py` | Validation suite (`python test_core.py`) |
| `METHODS_EXPLAINED.md` | Full math + plain-language explanation (shown in app) |

---

## What changed, and why it matters

### 1. `.ppd` reading was wrong for some file types

The old `read_ppd` always reshaped the data block into
`n_analog_signals + n_digital_signals` = 4 columns.

Per the [pyPhotometry spec](https://pyphotometry.readthedocs.io/en/latest/user-guide/importing-data/),
each 2-byte word holds a 15-bit analog sample (top bits) and one digital sample
(bottom bit), but the **frame width depends on acquisition mode**:

| File type | Words per frame | Layout |
|---|---|---|
| Pulsed, pyPhotometry **v1.1+** | 4 | ch1 LED-on, ch1 LED-off, ch2 LED-on, ch2 LED-off |
| Continuous mode, **or any pulsed file before v1.1** | 2 | ch1, ch2 (already dark-subtracted on-board) |

A fixed width of 4 is correct only for the first row. On a 2-word file it
interleaves two consecutive timepoints. Measured against synthetic files with
known ground truth (`test_core.py`, section 3):

- signal vs isosbestic correlation became **r = 1.0000** — the same quantity
  computed twice — where the true correlation was **−0.16**
- the sample count **halved**, so a 48 h recording read as 24 h

If all your files are v1.1+ pulsed, the old code was right *by coincidence* and
your existing results stand. If any are continuous-mode or pre-v1.1, results
from those files were meaningless. `photometry_core.read_ppd` now selects the
layout from the header `mode` and `version` fields, with a fallback if the
declared layout does not divide the data.

Two further fixes in the same function:

- **Header keys.** The spec names them `n_analog_channels` / `n_digital_channels`.
  The old code read `n_analog_signals` / `n_digital_signals`, which are the keys
  of the dictionary *returned* by `import_ppd`, not the keys *in the file header*.
  On a spec-compliant file this raises `KeyError`. Both spellings are now accepted.
- **`volts_per_division` indexing.** `vpd[min(i, len(vpd)-1)]` mapped the four
  columns to scale factors `[0, 1, 1, 1]`. The correct mapping is `[0, 0, 1, 1]` —
  both words of channel 1 share channel 1's scale factor. Column 1 (the 465 dark
  baseline) was being scaled with channel 2's value. Harmless when the two values
  are identical, wrong whenever they are not.

### 2. The `.ppd` and `.csv` routes disagreed

`.ppd` files went through `ana.py`; `.csv` files through `app.py`. They used:

| | `ana.py` (`.ppd`) | `app.py` (`.csv`) |
|---|---|---|
| Low-pass | 10 Hz | 1 Hz |
| Motion correction | global linear fit | 60 s rolling regression |
| F0 | 0.001 Hz low-pass of 465 | 10th percentile of **405** |
| dF/F units | ratio | percent |

The same animal gave different answers depending on which file you uploaded.
Everything now routes through `photometry_core.process_photometry`.

### 3. Three calculation errors in the CSV path

**F0 was taken from the isosbestic channel.**

```python
f0 = np.nanpercentile(uv_raw, 10)   # uv_raw is the 405 channel
dff = 100.0 * delta_f / f0
```

F0 is baseline fluorescence of the *signal*. Dividing by a percentile of the raw
405 trace divides by a different physical quantity on a different scale. On the
test dataset this inflated reported dF/F by **1.7×** (27% vs 16%). F0 is now
derived from the 465 channel, with selectable method (slow low-pass, rolling
percentile, session percentile, or median).

**The fit was computed on filtered traces but applied to raw ones.**

```python
uv_fit = slope * uv_raw + intercept   # slope came from uv_filt/sig_filt
delta_f = sig_raw - uv_fit
```

This reinjected all the high-frequency noise the filter had just removed —
**1.7× more noise** than intended. deltaF is now computed entirely from filtered
traces.

**Isosbestic fitting used OLS.** Real calcium transients are genuine 465-only
divergences from the control channel. OLS treats them as error and pulls the fit
toward them, subtracting part of the real signal away. The default is now IRLS
(Huber, tuning constant 1.345), following Keevers & Jean-Richard-dit-Bressel
(2025) *Neurophotonics* 12:025003, which found IRLS superior to OLS for exactly
this step. OLS remains selectable via `fit_method`.

### 4. Performance and numerical stability

- **Median filter.** `scipy.signal.medfilt` is O(n·k) and zero-pads its edges.
  On a 48 h recording at 130 Hz with a 1 s kernel that is ~3×10⁹ operations, and
  it corrupts the first and last half-kernel of the trace. Replaced with
  `scipy.ndimage.median_filter` (rolling histogram, edge reflection): **0.05 s
  for 2 h of data**.
- **The 0.001 Hz F0 filter.** A 3rd-order Butterworth designed that close to DC
  at full sampling rate sits pathologically near the unit circle and loses
  precision. `lowpass` now decimates, filters, and interpolates back when the
  normalised cutoff drops below 1e-3: numerically stable and **0.37 s for 6 h**.

### 5. z-scoring

The app z-scored over the whole session everywhere. That is reasonable for
describing the shape of one long trace, but it forces every session to mean 0 and
SD 1 *by construction*, so values are not comparable across sessions or animals,
and a session containing large transients inflates its own denominator.

`photometry_core.zscore` now supports three modes:

| Mode | Centre / spread from | Use for |
|---|---|---|
| `session` | whole recording | describing one long trace (previous behaviour, still the default) |
| `baseline` | a defined baseline epoch | **event-locked analysis** — units become "SDs from this animal's own resting state" |
| `robust` | median and 1.4826·MAD | when transients or artefacts would inflate the SD |

The lickometer peri-event analysis uses `baseline` by default.

> **Note on paper-matching.** These corrections are grounded in the pyPhotometry
> binary spec and the Keevers & Jean-Richard-dit-Bressel (2025) methods paper,
> both verified directly. Searches for the specific Lowell-lab and Knight-lab
> methods sections returned general methodology papers rather than the exact
> papers intended. If you identify the specific references, the parameters in
> `photometry_core.DEFAULTS` can be checked line by line against them.

### 6. Cropping with time re-zero (new)

Sidebar → **Crop recording and re-zero time**.

Crop a 48 h recording at `24:00:00` and what was hour 24 becomes `00:00:00`,
hour 25 becomes hour 1, and so on.

- The offset subtracted is **exactly the requested boundary**, not the timestamp
  of the first surviving sample. If acquisition starts a fraction of a second
  after the boundary, using the first sample would shift the axis by that
  fraction; using the boundary makes "crop at 24 h" land on a clean zero.
- The original timeline is preserved in `original_time_sec`.
- Event annotations and lick times shift with the crop.
- Downloads match what is on screen.
- **Recompute z-score on the cropped window** (default on): after cropping, the
  kept window *is* the recording, so its z-score should describe that window.
  Turn it off to keep whole-session statistics — appropriate when comparing a
  segment against statistics defined over the full session.

This is distinct from the pre-existing **Visual crop window**, which only changes
what is drawn and leaves the data untouched. Both are available.

### 7. Lickometer section (new)

Collapsed by default; opens from the sidebar. Photometry workflow is unchanged
for anyone not recording licks.

**Input:** a pyPhotometry digital input (`digital_1` / `digital_2`), or an
uploaded CSV (one timestamp per lick, or a time column plus a 0/1 state column).
The digital input shares the photometry clock, so no alignment step is needed.

**Analysis:** debounce → bout detection (configurable inter-bout gap and minimum
licks per bout) → lick raster and rate aligned under dF/F → peri-bout-onset
heatmap with mean ± SEM → CSV exports of bouts, lick times, and the peri-event
matrix.

**One subtlety worth knowing about.** Photometry exports at 1 Hz, but mice lick
at 6–10 Hz. A downsampled binary digital column collapses an entire ~30-lick bout
into a single rising edge. During development this produced **17 licks and 0
bouts** where the truth was 750 licks and 25 bouts.

Two fixes:

1. `ana.py` writes exact rising-edge times at the **full acquisition rate** to
   `event_times_digital_N.csv`, before any downsampling. The app prefers these.
2. Downsampling no longer averages digital columns — averaging a TTL turns a lick
   into a fraction and destroys the edge. It now counts edges per bin into
   `digital_N_pulse_count`, so lick counts survive intact.

If the app ever has to fall back to approximate timing, it says so in the UI
rather than letting you trust it silently.

Validated end-to-end against a synthetic 3 h recording with 750 licks in 25
bouts: **750/750 licks recovered, 25 bouts, 30.0 licks/bout, 7.15 Hz within-bout
rate**, peri-bout baseline 0.000 → **+5.00 SD** post-onset.

### 7b. Lick CSVs: any layout, auto-detected (new)

`lickometer.load_licks_csv()` works out the layout itself. It reads:

| Layout | Example |
|---|---|
| LIQ HD raw export | `record_type,pc_clock,elapsed_s,device_ms,channel,...` |
| One timestamp per lick | `lick_time` in s, ms, min, or clock times |
| Time + 0/1 state column(s) | one column per bottle |
| Time + per-bin lick counts | hourly or per-minute totals |

Multi-bottle files are split automatically, by a `channel`/`bottle` column or by
one state column per bottle. Files with an index column, extra metadata columns,
or unsorted rows are handled.

**Why this needed more than a column picker.** A file can offer several time
columns that disagree, and choosing wrongly does not raise an error — it
silently produces a nonsense recording. The LIQ HD export that prompted this
contains both `pc_clock` and `elapsed_s`, and `elapsed_s` is corrupt: across the
recording the clock advances 23.4 h while `elapsed_s` advances 82,343 h. Read
with `elapsed_s` the file becomes a 9-year recording at essentially zero licks
per hour.

So candidate time columns are cross-checked against each other. A wall clock is
trusted over a free-running counter, because a counter can drift, wrap or be
reset by firmware and a clock cannot, and the disagreement is reported in the UI
rather than resolved silently. Metadata columns (`peak`, `dc`, `duration_ms`,
`split`) are excluded outright so they can never be mistaken for a time axis.

Units are inferred from whether the resulting **duration** is physically
plausible, not from the typical gap between licks. Gap-based inference fails on
a sensor that double-counts contacts: gaps of a few ms read as a file in minutes,
and every timestamp gets multiplied by sixty.

Times stay absolute. Re-basing to the first lick would shift the whole raster by
however long the animal took to start drinking, quietly misaligning it against
the photometry. Only a wall-clock column is re-based, to the first timestamp,
since its epoch carries no meaning.

Two sanity checks run on load and surface as warnings: an implausibly low
licks-per-hour (usually the wrong time column), and a median inter-lick interval
faster than a mouse can lick (usually a sensor double-counting).

**Plotting.** `plot_licks`, `plot_licks_with_signal` and `plot_peri_event` build
the figures, so the same picture can be made from a script. Two details matter at
24 h scale: the raster is decimated above ~40k ticks (the rate trace below always
uses every lick, so nothing quantitative is lost), and bouts are shaded with one
NaN-separated filled trace rather than one layout shape each — 1,035 `add_vrect`
calls took minutes and left the browser crawling; the single trace draws in
0.13 s.

**Lick-only analysis.** The lickometer section used to live inside the
per-recording layout, so uploading a lick CSV with no `.ppd` or `.csv` beside it
produced an empty page. A lick file alone now gets its own section with the
raster, per-bottle summary, bouts and CSV downloads.

### 8. Appearance

The red-on-black multiselect chips came from three stacked `<style>` blocks
fighting each other with blanket `!important` rules. Two did the damage:

```css
section[data-testid="stSidebar"] *:not(svg):not(path) { color: #111827 !important; }
div[data-baseweb="select"] div { background-color: #ffffff !important; }
```

The first repainted every descendant in the sidebar, including chip text meant to
be light-on-coloured. The second matched *every* nested div inside a select — not
just the outer control — so tag chips, their remove buttons and the clear button
all lost their styling.

Replaced with `.streamlit/config.toml` for base theming (so Streamlit styles its
own widgets correctly) plus one scoped stylesheet in `theme.py` that sets the
chips explicitly and uses no universal selectors. Also: colour-blind-safe trace
palette (Okabe–Ito derived), consistent Plotly styling, and removal of three
duplicate function definitions (`finish_fig`, `make_overview_figure`,
`make_raw_independent_figure` were each defined twice — Python kept the second,
so the first copies were dead code).

---

## Validation

```bash
python test_core.py
```

Covers `.ppd` reading in both frame layouts, demonstration of the old reader's
failure mode, crop/re-zero arithmetic, low-cutoff filter stability, median filter
speed and edge behaviour, IRLS vs OLS slope recovery, all three z-score modes,
and a full pipeline run. All tests pass.

`python test_lick.py` covers lick CSV auto-detection: the real LIQ HD file, all
four layouts, unit inference, multi-bottle splitting, debounce, malformed files,
backward compatibility with the previous API, and the plotting helpers.

**Testing was against synthetic files with known ground truth, not real
recordings.** Before trusting any published number, re-run one file you know well
and compare. Expect dF/F magnitudes to be **smaller** now that F0 is correct —
that change is the fix working, not a regression.
