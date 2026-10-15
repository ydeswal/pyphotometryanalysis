# Updating the GitHub repository (15 Oct 2026, update b)

## Easiest: replace the whole folder

1. Unzip `pyphotometryanalysis_updated.zip`.
2. In your local clone, copy every file from it over the old ones. Show
   hidden files first (Mac: Cmd+Shift+. in Finder) so `.streamlit/` and
   `.gitignore` come along.
3. Delete these from the repository. They are not code, or are now replaced:

       pyphotometryanalysis_repo.tar.gz   (56 MB archive; re-cloned on every deploy)
       __pycache__/   .DS_Store   dot_streamlit/   config.toml (the one at the top level)

4. Commit and push:

       git add -A
       git commit -m "Side-by-side graphs, peaks, time ruler; cut memory use"
       git push

## Using the GitHub website instead

Upload these, all together (they depend on each other):

    app.py  graph_tools.py  photometry_core.py  ana.py  requirements.txt
    README.md  UPDATE_THESE_FILES.md  DEPLOY.md  test_graph_tools.py
    run_local.command  run_local.bat  run_local.sh  .gitignore

Then **Add file → Create new file**, name it `.streamlit/config.toml`, and
paste in the contents of the `.streamlit/config.toml` from the zip. Delete
the files listed in step 3 above.

Unchanged: lickometer.py, theme.py, convert_ppd.py, check_ppd.py,
test_core.py, test_lick.py, METHODS_EXPLAINED.md.

Do NOT also install the "graph workspace upgrade" (install_graph_workspace.py)
from the other proposal. Its features are included here, and its installer
would patch app.py a second time.

## After pushing

Streamlit Cloud redeploys on its own. If the page still says "Oh no", open
<https://share.streamlit.io>, then **⋮ → Reboot app** on
pyphotometrysweeneylab. A crashed app sometimes stays down until it is
rebooted.

## Check it worked

    python test_core.py          # ALL TESTS PASSED
    python test_graph_tools.py   # ALL TESTS PASSED

In the app, upload `m10_fasted-2026-10-14-102300.ppd`, then check:

- the **Display** buttons (One recording at a time / Side-by-side graphs)
  appear above the results;
- **Main dFF → ▲▼ Peaks** gives ceiling 20.15 % at 01:40:36 clock time;
- removing it moves the ceiling to 13.69 % at 08:10:28.
