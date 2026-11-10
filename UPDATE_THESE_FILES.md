# Update of 9 November 2026 (v12): FED health

Copy every file in this folder into your clone of `ydeswal/pyphotometryanalysis`, replacing the old ones,
then commit and push. Builds on v11 (the version on GitHub on 9 Nov, commit c2ae30e).

Changed: `fed3.py`, `fed_view.py`, `ai_chat.py`, `graph_tools.py` (one line), `test_fed3.py`, `README.md`,
this file. Adds the **FED health** view and reads everything FED3 controller 2.0 records (jams, JAMMED,
USB disconnects, blocked pokes, SD-card errors, restarts, email alerts); see README, "9 November 2026:
FED health". Check: `python test_fed3.py` and `python test_ai_chat.py` print ALL TESTS PASSED.

---

# Update of 9 November 2026 (v11)

Includes v7 to v10. This version fixes the AI not seeing a just-added event.
