# How this app turns light into neuron activity

This page explains every calculation the app performs, in two layers: the exact
math, and a plain-language version ("**In plain words**") that a high-school
student could follow.

**Primary reference.** Douglass AM, Kucukdereli H, Madara JC, ... Lowell BB
(2025). *Acute and circadian feedforward regulation of agouti-related peptide
hunger neurons.* Cell Metabolism 37:708–722. doi:10.1016/j.cmet.2024.11.009
(STAR Methods → "Fiber photometry recording analysis").

**Secondary reference** (cited by the paper as ref. 30, used for details the
paper does not state): Akam T & Walton ME (2019). *pyPhotometry: open source
Python based hardware and software for fiber photometry data acquisition.*
Sci Rep 9:3521, and the official pyPhotometry code (`tools/data_import.py`).

---

## The big picture

**In plain words.** A virus makes the neurons produce GCaMP, a protein that glows
brighter when the neuron is active (active neurons let calcium in, and calcium
makes GCaMP glow). An optic fiber shines blue light (465 nm) in and carries the
green glow back out to a light detector. The brighter the glow, the more active
the neurons.

The problem: the glow also changes for boring reasons. The mouse moves and the
cable bends, blood flow changes, and the dye slowly fades over hours
(photobleaching). So the fiber also shines violet light (405 nm). At 405 nm,
GCaMP's glow barely depends on calcium, so the 405 signal records the boring
changes only. We use it to subtract them out.

The pipeline, in order:

| Step | What it does | Formula |
|---|---|---|
| 0 | Read file, remove room light | $S = S_{on} - S_{off}$ |
| 1 | Median filter (remove spikes) | median of 5 neighbours |
| 2 | Low-pass filter (remove fast noise) | 10 Hz Butterworth, zero-phase |
| 3 | Fit 405 to 465 (motion model) | $\hat{U}(t) = a\,U(t) + b$ |
| 4 | Remove motion | $\Delta F = S - \hat{U}$ |
| 5 | Baseline | $F_0 =$ 0.001 Hz low-pass of $S$ |
| 6 | Normalise | $\Delta F/F_0$ |
| 7 | Standardise | $z = (x - \bar{x}) / s$ |
| 8 | Long-term trend | 30-min running mean |

Notation used below: $S(t)$ is the 465 nm signal, $U(t)$ is the 405 nm control
("UV"), $t$ is time, $f_s$ is the sampling rate (20 Hz for your files).

---

## Step 0. Reading the `.ppd` file

pyPhotometry saves a short text header (JSON) and then a long stream of
16-bit numbers. Each 16-bit number packs two things:

$$\text{word} = 2 \times \text{analog} + \text{digital}$$

The top 15 bits are the light reading; the bottom bit is the digital input
(e.g. a lickometer). The app separates them with bit operations:
`analog = word >> 1`, `digital = word & 1`.

Readings become volts by multiplying by `volts_per_division` from the header
(0.00010122 V for your file).

In pulsed mode (your mode is `2EX_1EM_pulsed`, version 1.1), each time point
stores **four** numbers: 465 with LED on, 465 with LED off, 405 with LED on, 405
with LED off. The LED-off reading measures stray light, so

$$S = S_{\text{LED on}} - S_{\text{LED off}}, \qquad U = U_{\text{LED on}} - U_{\text{LED off}}$$

This is exactly what the paper describes ("subtracting the baseline signal
before LED illumination reduced the effects of ambient light") and what the
official pyPhotometry importer does. The app also flags **clipping**: any
sample above 98% of the detector's maximum, using the same rule as the
official importer.

**In plain words.** Before each flash the machine measures how dark the room is.
Subtracting that is like weighing an empty cup first, so you only count the
water.

---

## Step 1. Median filter

Each point is replaced by the median (middle value) of itself and its
neighbours, using a window of $k = 5$ samples:

$$\tilde{S}(t_i) = \operatorname{median}\{S(t_{i-2}), \dots, S(t_{i+2})\}$$

The paper says data were "median-filtered" but gives no window size. 5 samples
is the pyPhotometry convention. At 20 Hz that is 0.25 s, short enough not to
blunt GCaMP6s transients. (The previous code used a 1-second window.)

**In plain words.** If one reading suddenly jumps because of an electrical glitch,
its four neighbours outvote it. Unlike an average, the median ignores a single
crazy value completely.

---

## Step 2. Low-pass filter (10 Hz)

A Butterworth filter keeps slow changes and removes fast ones. Its gain at
frequency $f$ is

$$|H(f)|^2 = \frac{1}{1 + (f / f_c)^{2n}}$$

with cutoff $f_c = 10$ Hz and order $n = 2$ (pyPhotometry's order; the paper
does not state one). The filter is run forwards and then backwards
(`filtfilt`), which cancels the time delay a one-way filter would cause, so
peaks are not shifted.

**Important for 20 Hz data.** The highest frequency a signal sampled at $f_s$
can contain is the Nyquist frequency, $f_s / 2$. At 20 Hz that is 10 Hz. So a
10 Hz low-pass on 20 Hz data has nothing to remove: it is the identity. The
app skips it and tells you so in "Processing notes". This is not a bug. The
paper's recordings were also 20 Hz.

**In plain words.** If you take 20 photos per second, you cannot see anything
flickering faster than 10 times per second. So a filter that removes "faster
than 10 per second" has no work to do.

---

## Step 3. Motion correction: fitting 405 onto 465

The paper: *"Motion correction was performed by finding a linear best fit to
the 405 nm signal and subtracting this from the 465 nm signal."*

We look for numbers $a$ (slope) and $b$ (intercept) that make the scaled 405
match the 465 as closely as possible:

$$\hat{U}(t) = a\,U(t) + b$$

"As closely as possible" means minimising the sum of squared differences
(ordinary least squares, OLS):

$$(a, b) = \arg\min_{a,b} \sum_t \big[S(t) - a\,U(t) - b\big]^2$$

Setting the derivatives to zero gives the closed-form answer:

$$a = \frac{\operatorname{cov}(U, S)}{\operatorname{var}(U)}, \qquad b = \bar{S} - a\,\bar{U}$$

The app reports the quality of the fit as $R^2$, the fraction of the 465
variance explained by the fitted 405:

$$R^2 = 1 - \frac{\sum_t [S(t) - \hat{U}(t)]^2}{\sum_t [S(t) - \bar{S}]^2}$$

The app's OLS is checked against `scipy.stats.linregress` in `test_core.py`
and agrees to 9 decimal places.

### Why a negative slope is rejected

Movement, cable bending and blood-flow changes alter how much light gets
through the fiber at **both** wavelengths **in the same direction**. Therefore a
genuine motion artefact always produces a **positive** slope $a$.

A negative slope means the regression found that 465 goes up when 405 goes
down. That is not motion. It happens when 465 contains a large real change
and 405 drifts slightly the other way. GCaMP is not perfectly
calcium-insensitive at exactly 405 nm (its true isosbestic point is a few nm
higher), so when calcium rises, the 405 glow dips a little. The regression
then multiplies that tiny noisy dip by a large negative number to "explain"
the real rise, and subtracting it deletes the biology and injects amplified
405 noise.

With **"Reject negative 405 slopes"** on (the default), the app applies the
constraint $a \ge 0$. The least-squares solution under that constraint, when
the unconstrained slope is negative, is

$$a = 0, \qquad b = \bar{S}$$

so nothing except a constant is subtracted, and a warning is displayed. Turn
the option off to get the literal unconstrained fit.

**In plain words.** The 405 light is the "bump detector". When the mouse bumps
its cable, both colours dim together, so we measure how much the 465 dims for
each unit the 405 dims and subtract that. If the math says "465 gets brighter
whenever 405 gets dimmer", that is not a bump: bumps make both dimmer. It is
real brain activity, and we must not subtract it.

---

## Step 4. ΔF (the motion-corrected signal)

$$\Delta F(t) = S(t) - \hat{U}(t)$$

in volts. Both terms use the filtered traces. (An earlier version of the CSV
route subtracted from the *unfiltered* trace, putting the noise back in.)

**In plain words.** What is left after removing the bump-related part is the
part that comes from the neurons.

---

## Step 5. F0, the baseline

The paper: *"F0 is the low-pass filtered 465 nm signal (0.001 Hz cutoff) of the
whole recording session."*

$$F_0(t) = \text{LowPass}_{0.001\,\text{Hz}}\big[S(t)\big]$$

0.001 Hz is one cycle every 1000 s ≈ 16.7 minutes. F0 therefore keeps only
changes slower than about a quarter of an hour, which over multi-day
recordings is mainly photobleaching.

Two numerical details matter.

**Edge padding.** A forward-backward filter needs data beyond each end of the
trace. The standard scipy padding is a handful of samples, under a second.
Against a 16-minute filter, that left the filter "holding" the first value of
the recording, so F0 equalled the first sample. The app now mirror-pads each
end by three filter periods ($3 / f_c$ = 50 minutes of mirrored data). The
result converges to what an ideal low-pass returns.

**Very low cutoffs.** Designing a 0.001 Hz filter directly at 20 Hz is
numerically fragile. The app first averages blocks of samples (block averaging
is itself a low-pass filter, so nothing fast leaks into the baseline), filters
the shorter series, then interpolates back.

**Short recordings.** If a recording is shorter than one filter period
(< 16.7 min), the 0.001 Hz component is simply the average, so
$F_0 \approx \bar{S}$. The app warns you. Your 3-minute M16 file is in this
case: $F_0 = 1.793$ V.

**In plain words.** F0 is "how bright it normally is". We find it by blurring the
recording so heavily that only the very slow trend survives, like looking at
a mountain range from far away: you see the overall slope, not the rocks.

---

## Step 6. ΔF/F0 (percent change)

$$\frac{\Delta F}{F_0}(t) = \frac{S(t) - \hat{U}(t)}{F_0(t)}$$

Stored as a ratio in the `_dFF` column (0.05 = 5%) and as percent in
`_dFF_percent` (×100). The graphs show percent.

**In plain words.** A 0.1-volt rise is big if normal brightness is 0.5 V (+20%)
and small if it is 5 V (+2%). Dividing by F0 turns volts into "percent brighter
than normal", so mice with more or less virus, or different fibers, can be
compared fairly.

---

## Step 7. Z-score

$$z(t) = \frac{x(t) - \bar{x}}{s_x}, \qquad x = \frac{\Delta F}{F_0}$$

where $\bar{x}$ is the session mean and $s_x$ the session standard deviation
($s_x = \sqrt{\tfrac{1}{N-1}\sum (x - \bar{x})^2}$). By construction, every
session has mean 0 and SD 1.

Optional modes (sidebar, lickometer) use a baseline window or the robust
version $z = (x - \text{median}) / (1.4826 \cdot \text{MAD})$, which is not
inflated by a few huge transients.

**In plain words.** z = 2 means "two typical wiggles above this mouse's average
today". It puts every recording on the same ruler.

---

## Step 8. Long-term traces (30-minute running mean)

For day-long recordings the paper plots a 30-minute running average:

$$\bar{z}_{30}(t) = \frac{1}{N_w}\sum_{|t_j - t| \le 15\,\text{min}} z(t_j)$$

The window is set in time (not sample count), so it is correct even with gaps.
Change it in **Processing method → Long-term running average** or in the
z-score tab.

**In plain words.** Like a weather report giving the weekly average temperature
instead of every minute's reading: it shows the trend (e.g. a daily rhythm)
and hides the jitter.

---

## Event-aligned measures from the paper

For feeding bouts the paper compares the mean z-score from 40 to 20 s before
bout start with the mean from 0 to 20 s after (for meals: 100–50 s before vs
0–50 s after):

$$\Delta z = \overline{z}_{\text{after}} - \overline{z}_{\text{before}}$$

(The methods write it as before − after but report decreases as negative
numbers, e.g. −0.21 for bouts and −0.75 for meals, so read it as
after − before.) The **Z-score range analysis** tab gives the mean z in any
windows you define, from which this difference can be taken.

---

## Alternative pipeline: pyPhotometry (Akam & Walton)

Selectable in **Processing method → Pipeline**. Instead of a low-pass baseline,
it fits a double-exponential bleaching curve to each channel:

$$B(t) = c + A_{fast}\,e^{-t/\tau_{fast}} + A_{slow}\,e^{-t/\tau_{slow}}$$

(fitted on 1-Hz averages, $60 \le \tau_{fast} \le 600$ s,
$600 \le \tau_{slow} \le 36000$ s, as in the official code). Then

$$S_{bc} = S - B_S, \quad U_{bc} = U - B_U, \quad \text{motion} = a\,U_{bc} + b$$

$$\Delta F = S_{bc} - \text{motion}, \qquad \frac{\Delta F}{F_0} = \frac{\Delta F}{B_S}$$

Fitting on bleach-corrected traces stops slow bleaching trends from dominating
the 405 regression. If the exponential fit fails to converge, the app falls
back to the Douglass pipeline and says so.

---

## What was changed, and why

| Problem | Effect | Fix |
|---|---|---|
| Negative 405 slope accepted | Real signal subtracted away, 405 noise amplified (your file: slope −17.8) | Slope constrained to ≥ 0, warning shown |
| F0 edge padding too short | F0 stuck at first sample (1.47 V instead of 1.79 V) | Mirror padding of 3 filter periods |
| Point-sampled decimation for F0 | Fast fluctuations aliased into baseline | Block-average before filtering |
| Robust (IRLS) fit as default | Not what the paper does | OLS default; IRLS optional |
| Filter order 3, 1-s median | Not the cited reference values | Order 2, 5-sample median (pyPhotometry) |
| dF/F ratio labelled "%" | 5% shown as 0.05% | Percent column, correct label |
| Hour-only time axis | 3-min file had no tick labels | Seconds / minutes / hours automatically |
| Always exported at 1 Hz | 3-min file reduced to 181 points | Auto: full rate for short files |
| Correction check vs raw 465 | Compared fit to a trace it was not fitted to | Compared to filtered 465 |
| 30-min mean on short files | All-NaN column | Time-based window, min 1 sample |

---

## Your M16 recording, by the numbers

- 20 Hz, 3 min 01 s, pulsed v1.1, no clipping.
- 465: 1.46 V for the first ~90 s, rising to 2.40 V by the end (+64%).
- 405: 0.730 V → 0.681 V (−7%). Correlation with 465: r = −0.90.
- Unconstrained 405 slope: −16.9 → rejected, set to 0.
- F0 = 1.793 V (session mean, because the file is shorter than 16.7 min).
- dF/F0: about −18% before the rise, about +34% over the last 30 s.
- z: −0.86 before, +1.61 at the end.

Because F0 is the whole-session mean, the pre-rise period sits below zero. For
short sessions built around a single event, express activity relative to a
pre-event baseline window instead (the paper's event-aligned approach).

---

## Checklist when you look at a new recording

1. **Slope a.** Should be positive. A clamped slope means the fit could not find
   a motion artefact; check the raw traces by eye.
2. **R².** High R² means much of 465 was explained by 405 (lots of motion was
   removed). Very low R² means little was removed, which is fine for a still
   animal.
3. **Clipping warning.** If present, lower the LED current; clipped data cannot
   be corrected.
4. **Duration warning.** Under ~17 min, F0 is just the mean.
5. **Raw channels tab.** Always look at the raw data. No pipeline replaces your
   eyes.

---

## Glossary

- **465 nm / 405 nm**: blue and violet excitation light. 465 reports calcium; 405 is the control.
- **Isosbestic point**: the wavelength at which GCaMP glows equally with or without calcium (~410–415 nm). 405 nm is close but not exact.
- **Nyquist frequency**: half the sampling rate; the fastest change the data can represent.
- **Low-pass filter**: keeps slow changes, removes fast ones.
- **Zero-phase filter**: run forwards and backwards so events aren't delayed.
- **OLS**: ordinary least squares, the standard "line of best fit".
- **R²**: fraction of variation explained by the fit (0 to 1).
- **Photobleaching**: dye slowly losing brightness under light exposure.
- **Z-score**: distance from the average, measured in standard deviations.
