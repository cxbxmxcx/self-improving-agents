# Self-Improving Agents: companion repository

Reference implementation of the architecture defined in [`SPEC.md`](SPEC.md),
companion to *Self-Improving Agents* by Micheal Lanham (Manning, in MEAP).

The repository implements **HelixAgent**, a general-knowledge agent with
composable skills that evolves chapter by chapter from a static v0 RAG-plus-ReAct
loop into a self-aware, self-improving production agent.

Spec version implemented: **0.3** (v0.2 artifact kinds and composition; v0.3 memory, scaffold, harness, and chapter renumbering amendments, SPEC §14).

## Layout

```
helix/                  framework modules (Artifact, Trajectory, Agent, ...)
helix_web/              the Helix Observatory: FastAPI + vanilla JS dashboard over run archives
chapters/ch02/          getting started: HelixAgent v0 and v1, SPO offline loop
chapters/ch03/          measuring and searching: ground-truth signals, SPO, the cost ladder
chapters/ch04/          evolutionary search: GEPA, DGM, the lineage dashboard
chapters/ch05/          value-based memory learning with a studio support walkthrough
agents/                 example agents (helpdesk, researcher, revenue)
chapter_appendices/     appendix A (setup), appendix B (RAG), ch3 v0 scaffolding
ingestion/              PDF -> LanceDB ingestion pipeline
data/helix_corpus.lance the shipped corpus (built from ingestion/pdfs/)
docs/architecture/      editable architecture diagram sources (drawio)
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

## Run the baseline agent

```
python chapters/ch02/01_helixagent_v0.py
```

The baseline agent (RAG + ReAct, fixed system prompt, no improvement) is
Chapter 2's first listing; edit the `question` variable in `main()` to ask
your own. Its building blocks live in `chapters/ch02/agent_v0.py`, the
tool-description scaffolding (demonstrated in Chapter 7) extends them in
[`chapter_appendices/getting_started/`](chapter_appendices/getting_started/),
and the chapter walkthrough is
[`chapters/ch02/README.md`](chapters/ch02/README.md).

## Run the tests

```
pytest
```

## Talk to an agent

```
streamlit run helix/chat_ui/app.py
```

A production-style chat UI. Sidebar picks the agent (from `agents/*.py`),
sets identity (session_id / user_id / org_id), and points at an archive so
the agent uses the current champion prompt. Thumbs / regenerate / copy
buttons under each agent response feed the FeedbackStore. Default agent is
`agents/helpdesk.py`: a RAG assistant with episodic, semantic, and
procedural memory.

## Explore a run in the Observatory

The Helix Observatory is the repo's web dashboard: a FastAPI server with a
hand-built frontend that reads any `*.sqlite` archive a chapter run produced.
Launch it and open http://localhost:8610:

```
python helix_web/run.py
```

It discovers every archive under `chapters/*/runs/` and shows, per archive:

- **Overview**: champion, score range, by-method breakdown, failure histogram
- **Champion**: best versus live version, with a human promote gate
- **Lineage**: the artifact family tree, colored by Search, gold crown on the top score, SVG export
- **Compare**: side-by-side content diff plus measurement history for any two versions
- **Verdicts**: every per-question judge decision, filterable, with a preference heatmap
- **Replay**: step-by-step trajectory inspection and cross-version answer diffs
- **Rounds**: score progression per round, or per version when no round log exists
- **Lineage compare**: two or more archives' trees side by side, the search-method showdown

The Observatory also streams live: leave the Lineage panel open while a loop
runs and new versions animate into the tree as they are recorded. The older
Streamlit dashboard (`streamlit run helix/dashboard/app.py`) still works but
is superseded by the Observatory.

## What this repo is not

The `helix/` framework deliberately ships none of the following; SPEC §12's
rule is "the framework emits events; consumers render." The chat UI, the
Observatory, and the chapter scripts are such consumers, living beside the
framework rather than inside it.

- No UI inside the framework. `helix_web/` and `helix/chat_ui/` consume helix as a library.
- Not a model gateway. LiteLLM handles routing across providers.
- Not a vector database. LanceDB is the local store; readers can swap.
- Not a workflow orchestrator. The agent loop is the loop.
- Not a fine-tuning toolkit. The book's thesis is strict-external-to-the-model.

See [SPEC.md §12](SPEC.md#12-non-goals) for the full non-goals list.
