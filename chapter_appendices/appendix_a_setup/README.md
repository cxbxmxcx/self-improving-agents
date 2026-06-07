# Appendix A: Setting up the source code

The README file has been the front door to source code since the early days of
software distribution, when a plain text note shipped alongside the source told
you how to build the thing before you could run it. The convention stuck because
the order never really changed: read first, set up, then run. Think of this
appendix as that note for the book's companion repository.

Every chapter in this book leans on that repository, and the fastest way to
learn the patterns is to run them. This appendix gets you from a fresh clone to
a working HelixAgent that answers a question against the corpus. The setup is
deliberately plain: clone, make an environment, add one API key, and run.

As a general rule, you only have to do this once. The corpus ships with the
repo, so you are not waiting on a long ingestion step before you can run
anything. If you would rather skim the high-level version, the repository root
[`README.md`](../../README.md) is the short form; this appendix is the version
with the reasoning and the things that trip people up.

## A.1 What you'll need

A few prerequisites before you start. None of them are exotic, and most
machines built in the last few years already have them.

- **Python 3.11 or newer.** The repo is developed and tested on 3.13; anything from 3.11 up should work.
- **Git.** To clone the repository and to pull updates as the book progresses.
- **An LLM provider key.** The Chapter 2 scripts default to Anthropic (Claude), but LiteLLM lets you point at OpenAI, Google, Groq, Mistral, and others.
- **About 1 GB of free disk.** Mostly the embedding model (~440 MB) and the shipped corpus index.

Remember, you do not need a GPU. The retrieval embedding model runs on CPU, and
the agents themselves call a hosted model over the network.

## A.2 Get the source

Clone the repository and step into it.

```
git clone https://github.com/cxbxmxcx/self-improving-agents.git
cd self-improving-agents
```

That directory is your working root for the rest of the book. Every command in
every chapter assumes you are running it from there.

## A.3 Create an environment and install

Make a virtual environment so the book's dependencies stay isolated from the
rest of your system. The activation command differs by platform; everything
after that is the same.

```
python -m venv .venv

.venv\Scripts\activate            # Windows (PowerShell)
# source .venv/bin/activate       # macOS / Linux

pip install -r requirements.txt
```

The install pulls LiteLLM, Pydantic, Instructor, LanceDB, sentence-transformers,
and the Streamlit stack used by the dashboards. The first install is the slow
one because sentence-transformers brings in PyTorch. Subsequent runs reuse what
pip already cached.

## A.4 Add your API key

Copy the sample environment file and fill in the provider you intend to use. The
`.env` file is gitignored, so your keys never end up in version control.

```
cp .env.sample .env               # macOS / Linux
# copy .env.sample .env           # Windows
```

Open `.env` and set the key for your provider. For the Chapter 2 defaults, that
is Anthropic.

```
ANTHROPIC_API_KEY="sk-ant-..."
HF_HUB_OFFLINE=1
```

The `HF_HUB_OFFLINE=1` line is optional but recommended. It tells Hugging Face
to skip a network check every time a retrieval index opens, which silences a
warning and speeds up local runs since the embedding model is already cached.
Every script calls `load_env()` at startup, so once the file is in place the
keys load automatically.

## A.5 The retrieval corpus

HelixAgent answers from a document corpus, stored as a LanceDB index at
`data/helix_corpus.lance/`. The book ships this index already built, from a set
of papers in the agent literature, so the chapter scripts work out of the box.
For most readers there is nothing to do here.

You only rebuild when you want HelixAgent to retrieve from your own documents.
Drop PDFs into `ingestion/pdfs/` and run the build, which extracts text, chunks
it, embeds with `BAAI/bge-base-en-v1.5`, and writes the index.

```
python ingestion/build_index.py
```

The first build downloads the ~440 MB embedding model into your local Hugging
Face cache; later builds are offline. If you want the exact paper set the book
uses, `python ingestion/download_corpus.py` fetches it into `ingestion/pdfs/`
before you build. See [`ingestion/README.md`](../../ingestion/README.md) for the
details.

## A.6 Verify the install

Run the baseline agent against a single question. This is the v0 agent: RAG plus
ReAct, a fixed prompt, no improvement yet.

```
python chapters/ch02/01_helixagent_v0.py
```

You should see the agent retrieve a few passages and return an answer, followed
by a short trajectory summary. If that works, your environment, your key, and
your corpus are all wired correctly. To run the spec-conformance tests as a
fuller check, use pytest.

```
pytest
```

The test suite exercises the framework against the corpus, so the first run
triggers the embedding-model download if you skipped the corpus step above.

## A.7 How the repository is wired

One thing that surprises people: this repo is not an installable package, and
there is no `pip install -e .`. Instead the repository root is placed on the
Python path, by `conftest.py` for tests and by a small shim at the top of each
script. The payoff is that any file can write `from helix.agent import Agent`
with no build step, which keeps the setup story to "clone and run."

The framework itself lives under `helix/`, and each chapter's runnable code
lives under `chapters/chNN/`. The book's running project, HelixAgent, evolves
across those chapters while the framework primitives stay stable.

| Path | What's there |
|------|--------------|
| `helix/` | Framework modules: Artifact, Trajectory, Agent, Signal, Search, Archive |
| `chapters/ch02/` | HelixAgent v0 and v1, the judge, SPO, the offline loop |
| `ingestion/` | PDF to LanceDB ingestion pipeline |
| `data/helix_corpus.lance/` | The shipped retrieval corpus |
| `tests/` | Spec-conformance tests |
| `SPEC.md` | The architectural specification, the source of truth |

## A.8 Running the chapter code

Chapter code examples are numbered to show the order you read them in. A file
named `01_helixagent_v0.py` is the first listing in Chapter 2, `02_...` is the
second, and so on. Run any of them directly from the repository root.

```
python chapters/ch02/01_helixagent_v0.py
python chapters/ch02/05_spo_offline_loop.py
```

Not every file in a chapter folder is a numbered listing. Shared helpers like
`agent_v0.py` and infrastructure like `eval_harness.py` carry plain names
because you import them rather than read them top to bottom. Each chapter folder
has its own `README.md` that maps the sections to the scripts; start with
[`chapters/ch02/README.md`](../../chapters/ch02/README.md).

## A.9 The dashboards (optional)

Once you have run an improvement loop, two Streamlit apps help you see what
happened. Neither is required to follow the book, but both make the abstract
loop concrete.

```
streamlit run helix/dashboard/app.py     # inspect the archive: lineage, verdicts, rounds
streamlit run helix/chat_ui/app.py        # talk to an agent backed by the champion prompt
```

The dashboard reads the archive that the chapter scripts write to
`chapters/ch02/runs/`, so run a chapter script at least once before you open it.

## A.10 Troubleshooting

The handful of issues readers actually hit, and the fix for each.

| Symptom | Cause and fix |
|---------|---------------|
| `FileNotFoundError: No corpus found at data/helix_corpus.lance` | The index is missing. Run `python ingestion/build_index.py`, or pull the shipped index from the repo. |
| `AuthenticationError` or 401 from the model | No provider key, or the wrong one. Confirm `.env` has the key for the model the script uses, and that you copied it from `.env.sample`. |
| A warning about unauthenticated Hugging Face Hub requests | Harmless. Set `HF_HUB_OFFLINE=1` in `.env` to silence it, since the model is already cached. |
| `ModuleNotFoundError: No module named 'helix'` | You are not at the repository root, or the venv is not active. `cd` to the repo root and re-activate `.venv`. |
| A `_distutils_hack` line on startup | Harmless venv noise from setuptools. It does not affect the run. |
| The first run is slow | The embedding model is downloading (~440 MB) on first use. Later runs are offline and fast. |

## Where to go next

You now have a working agent and the tools to watch it improve. Head to Chapter
2 and run the listings in order; the costs per run are small, and
[`chapters/ch02/README.md`](../../chapters/ch02/README.md) gives a rough
per-round estimate so there are no surprises. Start with `01_helixagent_v0.py`,
get a feel for the baseline, and let the rest of the chapter take it from there.
