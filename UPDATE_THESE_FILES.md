# Which files to upload to GitHub

Upload ALL of these (they depend on each other):

    app.py
    ana.py
    photometry_core.py        <- all the maths
    METHODS_EXPLAINED.md      <- shown in the "Methods & math" tab
    lickometer.py             (unchanged, included for completeness)
    theme.py                  (unchanged)
    requirements.txt
    README.md
    test_core.py
    test_lick.py              (unchanged)
    .gitignore
    .streamlit/config.toml    (unchanged)

`.streamlit` starts with a dot, so it is hidden on Mac/Windows. A visible copy
is included as `dot_streamlit/config.toml`. On GitHub's web uploader, create
it with **Add file -> Create new file**, type `.streamlit/config.toml` as the
name, and paste the contents.

Do NOT upload `__pycache__/`, `.DS_Store`, or `.ppd` data files.

## Check it worked

Upload the M16 .ppd. Above the graphs you should see:

    Pipeline: Douglass 2025 | 405->465 slope a: 0 (raw -16.9) | Fit R^2: 0.000 | F0: 1.793 V

plus two orange warnings (negative slope, short recording), the x-axis in
seconds, and a "Methods & math" tab.
