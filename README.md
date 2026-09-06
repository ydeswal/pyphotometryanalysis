# pyPhotometry Analysis (Sweeney Lab)

Streamlit app for analysing pyPhotometry `.ppd` and `.csv` fiber photometry
recordings: motion correction, dF/F, z-scoring, long-timescale visualisation,
and lickometer analysis.

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## File layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `ana.py` | Batch `.ppd` command-line pipeline (called by `app.py`) |
| `photometry_core.py` | **All maths.** Single source of truth |
| `lickometer.py` | Lick ingestion, bout detection, peri-event alignment |
| `theme.py` | Styling and Plotly defaults |
| `.streamlit/config.toml` | Base theme |
| `test_core.py` | Validation suite (`python test_core.py`) |

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

**Testing was against synthetic files with known ground truth, not real
recordings.** Before trusting any published number, re-run one file you know well
and compare. Expect dF/F magnitudes to be **smaller** now that F0 is correct —
that change is the fix working, not a regression.
