# Helix Observatory

The Observatory is the web dashboard for Helix run archives. It is a FastAPI
server plus a hand-built HTML/CSS/JS frontend, with d3 as the only vendored
library and no build step, so every file it serves is readable as written.

The framework itself ships no UI; SPEC section 12 says the framework emits
events and consumers render. The Observatory is such a consumer, importing
`helix` as a library and reading archives strictly read-only.

## Run it

```
python helix_web/run.py
```

Then open http://localhost:8610. Useful flags: `--port`, `--host`,
`--archive PATH` to register an archive outside `chapters/*/runs/`, and
`--poll-ms` for the live watcher cadence.

## What it shows

Every chapter archive is discovered automatically and every view is a URL,
so any panel, node, or comparison can be bookmarked or shared. The seven
panels mirror the improvement loop: Overview, Champion, Lineage, Compare,
Verdicts, Replay, and Rounds, plus a cross-archive Lineage compare.

The Lineage tree is the centerpiece: families laid out as tidy trees, nodes
colored by the Search that produced them, a gold crown on the top measured
score, and dashed edges for a crossover's second parent. Export SVG produces
a print-ready figure, which is how the book's dashboard figures are made.

## Live mode

Leave a panel open while a loop runs and the header pill turns live; new
versions animate into the tree as the loop records them. The server polls
the archive's rowid watermarks (the event bus is in-process to the loops),
flips the file to WAL on first watch, and streams typed SSE events with a
cursor that doubles as the reconnect position.

## Writes

The Observatory performs exactly two writes, both explicit: promoting a
version to live (the human gate from chapter 2) and purging failed
trajectories. Everything else opens the archive read-only, so browsing a
run can never mutate it.

## Tests

```
python -m pytest tests/test_web_*.py tests/test_web_panels_data.py
```

The fixture in `tests/conftest.py` builds a small archive covering every
shape the panels render: spo, gepa, crossover, dgm lineages, a promotion,
and ok, zero-step, and errored trajectories.
