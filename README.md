# Self-Improving Agents — companion repository

Reference implementation of the architecture defined in [`SPEC.md`](SPEC.md),
companion to *Self-Improving Agents* by Micheal Lanham (Manning, in MEAP).

The repository implements **HelixAgent**, a general-knowledge agent with
composable skills that evolves chapter by chapter from a static v0 RAG-plus-ReAct
loop into a self-aware, self-improving production agent.

Spec version implemented: **0.1**.

## Layout

```
helix/                  framework modules (Artifact, Trajectory, Agent, ...)
chapters/ch02/          HelixAgent v0 and v1 scripts
ingestion/              PDF → LanceDB ingestion pipeline
data/helix_corpus.lance the shipped corpus (built from ingestion/pdfs/)
tests/                  spec-conformance tests
SPEC.md                 architectural specification (source of truth)
DESIGN_NOTES.md         the why behind the spec
```

This is not a Python package. There is no `pip install -e .`. The repo root is
on `sys.path` (via `conftest.py` for tests and a small shim in each chapter
script), so any code can `from helix.agent import Agent` directly.

## Setup

```
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

Set whichever LLM provider key you want LiteLLM to use:

```
$env:OPENAI_API_KEY = "..."        # PowerShell
# export OPENAI_API_KEY=...        # bash
```

## Build the retrieval corpus

Drop PDFs into `ingestion/pdfs/`, then:

```
python ingestion/build_index.py
```

The first run downloads the embedding model (`BAAI/bge-base-en-v1.5`, ~440MB)
into your local Hugging Face cache. Subsequent runs are offline.

The result is a LanceDB index at `data/helix_corpus.lance/`. Chapter scripts
read from there.

## Run HelixAgent v0

```
python chapters/ch02/helixagent_v0.py
python chapters/ch02/helixagent_v0.py --question "What is GEPA?"
```

See [`chapters/ch02/README.md`](chapters/ch02/README.md) for what v0 is and
what v1 adds.

## Run the tests

```
pytest
```

## Explore a run in the dashboard

After running any chapter script that populates `chapters/ch02/runs/helix_archive.sqlite`,
launch the Helix dashboard:

```
streamlit run helix/dashboard/app.py
```

The dashboard mines the archive, trajectory cache, and round log to show:

- **Overview** — current champion, score range, by-method breakdown
- **Lineage** — interactive tree of every artifact, colored by which Search produced it
- **Compare** — side-by-side prompt diff between any two artifacts, with measurement history
- **Verdicts** — every per-question judge decision, filterable by version, band, role
- **Replay** — full trajectory inspection for any cached (artifact, question) pair, with cross-trajectory diff
- **Rounds** — score progression over rounds, promotion markers, cost timeline

Point the sidebar at a different archive to compare runs side by side.

## What this repo is not

- Not a UI, dashboard, or operator interface.
- Not a model gateway. LiteLLM handles routing across providers.
- Not a vector database. LanceDB is the local store; readers can swap.
- Not a workflow orchestrator. The agent loop is the loop.
- Not a fine-tuning toolkit. The book's thesis is strict-external-to-the-model.

See [SPEC.md §12](SPEC.md#12-non-goals) for the full non-goals list.
