# Which files to upload to GitHub (15 Oct 2026 update)

Changed or new in this update, upload these:

    photometry_core.py        <- reads PC-timestamped .ppd files + clock model
    ana.py                    <- clock_time / clock_unix columns, gap-aware times
    app.py                    <- clock on hover + summary, time-axis option,
                                 lick CSVs aligned by clock
    convert_ppd.py            NEW: PC-timestamped .ppd -> standard .ppd
    check_ppd.py              <- reports layout, clock checks, verdict
    test_core.py              <- sections 19-23
    README.md
    METHODS_EXPLAINED.md      <- shown in the "Methods & math" tab
    UPDATE_THESE_FILES.md

Unchanged: lickometer.py, theme.py, requirements.txt, test_lick.py,
.streamlit/config.toml.

`photometry_core.py`, `ana.py` and `app.py` depend on each other, so upload
all three together. The app runs `ana.py` in a separate process, and an old
`photometry_core.py` makes it fail with an AttributeError.

Do NOT upload `__pycache__/`, `.DS_Store`, or `.ppd` data files.

## Check it worked

    python3 test_core.py      # ALL TESTS PASSED

Upload `m10_fasted-2026-10-14-102300.ppd`. Above the graphs you should see:

    Pipeline: Douglass 2025 | slope a: 2.66 | Fit R^2: 0.892 | F0: 1.129 V | Duration: 23:55:48
    Wed 2026-10-14 10:23:00 -> Thu 2026-10-15 10:18:48 local time · PC clock ...

with no clipping warning. Hovering a trace shows
`Time: hh:mm:ss (clock MM-DD hh:mm:ss.fff)`.
