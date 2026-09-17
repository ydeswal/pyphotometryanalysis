# Two files must be uploaded together

The error

    module 'lickometer' has no attribute 'load_licks_csv'

means `app.py` was updated on the server but `lickometer.py` was not. The new
`app.py` calls functions that only exist in the new `lickometer.py`.

## Upload both

    app.py
    lickometer.py        <-  this is the one that was missed

Nothing else has to change for the lickometer fix. `photometry_core.py`,
`ana.py`, `theme.py` and `requirements.txt` are included in this zip unchanged
from your last deploy, so copying the whole folder is also safe.

## Check it worked

The Lickometer section should show a grey caption like

    Detected LIQ HD raw export - time from pc_clock - 17,929 licks across 1 bottle(s) - 23.44 h

If instead you see a red box saying `lickometer.py` on the server is out of
date, the upload did not take. Two things to check:

1. On GitHub, open `lickometer.py` and search the page for `load_licks_csv`.
   If it is not there, the file did not upload.
2. Make sure no `__pycache__/` folder or `lickometer.pyc` was committed. A
   stale compiled file can shadow the source. Delete it from the repo if present
   and add `__pycache__/` to `.gitignore`.

## Version pinning

`lickometer.py` now carries `__version__ = "2.0"` and `app.py` checks it at
startup. If the two ever drift apart again you get a plain message naming the
file to update, instead of an AttributeError from somewhere in the middle of the
page.
