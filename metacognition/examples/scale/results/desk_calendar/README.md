# The calendar desk's paid run, November 11, 2026

The results that section 11 of [SCALE_TEST.md](../../../../SCALE_TEST.md) reports. The recordings, about 76 MB, are kept compressed outside git. With them unpacked into `examples/scale/runs/desk_calendar/recorded`, each part replays with no key, for example `python -m examples.scale.desk_run full --output <folder>`, and `python -m examples.scale.desk_report` computes every measure again.

| File | What it holds |
|---|---|
| `report.json` | Every measure and criterion, as `desk_report.py` computed them |
| `alone.json`, `loop.json`, `full.json` | Each assistant's requests, a row each, with its nights, its learning log and its spend |
| `searches.json` | The five search repeats |
| `check.json` | The live check of the wiring |
| `figures/` | The data behind the results page's charts and tables |
| `day17/judged_versions.json` | Each same-time request from day 14 on: the function version the agent was offered, the days as it wrote them, whether it was served right, and each version's verdict with the agent's own arguments, with the request's canonical arguments, and with the days written in full |
| `day17/functions.md` | Both versions of `find_common_starts` as the builder wrote them, and the line where they differ |
