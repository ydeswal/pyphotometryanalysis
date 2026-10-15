# Running and publishing the app

## On your own computer (recommended for long or many recordings)

Double-click `run_local.command` (Mac), `run_local.bat` (Windows) or run
`bash run_local.sh` (Linux). See the README section "Run it on your own
computer". There is no memory cap locally, and uploads up to 4 GB are allowed.

## The hosted site (Streamlit Community Cloud)

The site redeploys automatically whenever `main` on GitHub changes. Limits
worth knowing:

- **2.7 GB of memory** for the whole app, shared by everyone using it at that
  moment. One 48 h, 130 Hz recording now peaks around 1 GB during analysis. Two
  people analysing large files at once can still hit the limit. Use the local
  launcher for those.
- **Uploads up to 1 GB per file** (set in `.streamlit/config.toml`, which
  must be in a folder called `.streamlit` with the dot).
- If the page shows **"Oh no. Error running app"**: open
  <https://share.streamlit.io>, then **⋮ → Reboot app**. Use **⋮ → Logs** to
  see why it stopped.

## Publishing changes

    git add -A
    git commit -m "describe the change"
    git push

Work on a branch first if you want to check a change before the live site
picks it up. The site follows whichever branch it was deployed from.
