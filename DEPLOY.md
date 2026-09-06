# How to publish this to GitHub

I can't push for you — I have no credentials for `ydeswal/pyphotometryanalysis`,
and you shouldn't paste an access token into a chat. Here are two routes; pick
whichever you prefer. Both take a couple of minutes.

---

## Route A — you already have the repo cloned locally (recommended)

1. Download `pyphotometryanalysis_repo.tar.gz` and unpack it:

   ```bash
   tar -xzf pyphotometryanalysis_repo.tar.gz
   ```

   You'll get a `pyphot/` folder containing every file plus a git repo with the
   change already committed.

2. Copy the files into your clone:

   ```bash
   cd /path/to/your/pyphotometryanalysis

   cp /path/to/pyphot/app.py .
   cp /path/to/pyphot/ana.py .
   cp /path/to/pyphot/photometry_core.py .
   cp /path/to/pyphot/lickometer.py .
   cp /path/to/pyphot/theme.py .
   cp /path/to/pyphot/requirements.txt .
   cp /path/to/pyphot/test_core.py .
   cp /path/to/pyphot/README.md .
   cp /path/to/pyphot/.gitignore .

   mkdir -p .streamlit
   cp /path/to/pyphot/.streamlit/config.toml .streamlit/config.toml
   ```

3. Check it runs before you push:

   ```bash
   pip install -r requirements.txt
   python test_core.py          # should print ALL TESTS PASSED
   streamlit run app.py
   ```

4. Commit and push:

   ```bash
   git checkout -b fix-calculations-and-lickometer
   git add -A
   git commit -m "Fix photometry calculations, add crop/re-zero and lickometer, redesign UI"
   git push -u origin fix-calculations-and-lickometer
   ```

   Then open a pull request on GitHub, or push straight to `main` if you'd
   rather skip the PR:

   ```bash
   git checkout main
   git merge fix-calculations-and-lickometer
   git push origin main
   ```

**Working on a branch first is worth it here.** Your Streamlit app redeploys
automatically from whatever branch it tracks, so pushing to `main` replaces the
live site immediately. A branch lets you check it first.

---

## Route B — upload through the GitHub web interface

If you'd rather not use the command line:

1. Download the individual files from this conversation.
2. Go to https://github.com/ydeswal/pyphotometryanalysis
3. **Add file → Upload files**, drag in `app.py`, `ana.py`,
   `photometry_core.py`, `lickometer.py`, `theme.py`, `requirements.txt`,
   `test_core.py`, `README.md`. Commit.
4. The `.streamlit/config.toml` needs its folder created explicitly. Use
   **Add file → Create new file**, type `.streamlit/config.toml` as the filename
   (typing the `/` creates the folder), paste the contents of
   `dot_streamlit/config.toml`, and commit.

> The file is delivered here as `dot_streamlit/config.toml` because folders
> beginning with a dot are hidden on most systems and easy to lose. It must end
> up at `.streamlit/config.toml` in the repo — the leading dot matters, or
> Streamlit won't find the theme and the styling will look wrong.

---

## After it deploys

The app redeploys automatically within a minute or two. Check
https://pyphotometrysweeneylab.streamlit.app/

Then, before trusting any number:

1. **Re-run one recording you know well** and compare against your previous
   output. dF/F magnitudes will be *smaller* now that F0 comes from the 465
   channel rather than the 405. That's the fix working, not a regression.

2. **Check which `.ppd` format your files are.** Run this on one file:

   ```python
   import photometry_core as pc
   d = pc.read_ppd("your_file.ppd")
   print(d["mode"], d["version"], d["frame_layout"])
   ```

   - `pulsed_v1.1_led_on_off` → the old code happened to read these correctly,
     so your previous results were structurally fine (the F0 and filtering
     fixes still change the numbers).
   - `one_word_per_channel` → the old code was misreading these badly.
     Anything previously analysed from files like this needs redoing.

3. **If you use the lickometer with `.ppd` digital inputs**, re-run the analysis
   rather than reusing old processed CSVs — the full-rate lick timestamps are
   only written by the new `ana.py`.

---

## If something breaks

Roll back with:

```bash
git revert HEAD
git push
```

The old behaviour is one commit away at all times.
