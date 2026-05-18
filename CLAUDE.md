# CLAUDE.md

Persistent context for Claude Code sessions on this repository.

This file is read on every session. Keep it short, durable, and prescriptive.

---

## What this repo is

This is the companion repository for **Self-Improving Agents** by Micheal Lanham (Manning, in MEAP), a book about LLM agents that measurably improve in production without weight updates. The repository implements the architecture defined in `SPEC.md`. The book references this repository the way *Designing Data-Intensive Applications* references real systems: the architecture is taught in the book, the code is where curious readers verify the patterns.

The running project across the book is **HelixAgent**, a general-knowledge agent with composable skills that evolves chapter by chapter from a static v0 RAG-plus-ReAct loop into a self-aware, self-improving production agent.

## Source-of-truth files

Read these before making non-trivial decisions:

- `SPEC.md` — the architectural specification (10 primitives, the agent loop, the hook system, the composition map). Treat as the source of truth.
- `DESIGN_NOTES.md` — the *why* behind the architecture. Read when a spec section seems arbitrary; the reasoning is there.
- `BOOK_PROPOSAL.md` (when added by author) — the book's audience, scope, table of contents, and chapter goals.

## Non-negotiables (from the spec)

Do not weaken these without an explicit author decision:

1. **Artifacts are immutable.** Mutations always produce new versions with a parent pointer. Never edit an existing version in place.
2. **Every artifact has a parent.** Genesis artifacts have `parent_id = None`. Hand-authored variants use `created_by = "human"`.
3. **Signals return `GapMeasurement`.** One union shape across all signal families (score, preference, feedback, confidence). Do not introduce signal-family-specific return types.
4. **Searches expose `propose` and `select`.** Async iterator for propose, single artifact result for select. Do not collapse them.
5. **The agent loop does not change after Chapter 2.** New capabilities are added as hooks, signals, search methods, or artifact kinds. Never as new agent loop branches.
6. **Hooks fire on the canonical points listed in Section 6.2 of the spec.** Adding new hook points requires a spec amendment.
7. **No bundled UI, model gateway, vector database, or workflow orchestrator.** These are non-goals (Section 12 of the spec).
8. **No fine-tuning, RLHF, or weight updates anywhere in this codebase.** The book's thesis is strict-external-to-the-model.

## The base stack

- **Python 3.11+** with type hints throughout. Use `Protocol` for primitive interfaces.
- **LiteLLM** for model-agnostic LLM calls. Do not bind to one provider.
- **Pydantic v2** for typed inputs, outputs, state, and schemas. Tool argument schemas derive from type hints via `create_model`.
- **Instructor** for typed Pydantic outputs with retry-on-validation.
- **MCP Python SDK** for MCP tool integration.
- **OpenTelemetry** for the observability bus. Emit OTel spans and the typed event schema from spec Section 10.1.
- **Async-first.** Every primitive interface is async. Sync wrappers may be provided for convenience.

## House style

Any prose, docstrings, README content, or commentary generated for this repo follows these rules:

- **Paragraphs are 3 sentences.** Hard rule. Break longer paragraphs at natural conceptual seams.
- **No em-dashes.** Use commas, semicolons, or colons.
- **Markdown only for documents.** No .docx unless explicitly requested by the author.
- **Mermaid for diagrams.** All architecture diagrams in Mermaid format.
- **Code comments explain intent, not mechanics.** "Why" not "what."
- **Test names are sentences.** `test_archive_records_parent_pointer_on_mutation` not `test_archive_1`.

## What to do on first session

If this is the first time touching the repo:

1. Read `SPEC.md` end to end. It's roughly 5,000 words; budget 20-30 minutes.
2. Skim `DESIGN_NOTES.md` for the reasoning behind controversial-looking decisions.
3. Propose a repository structure that implements the primitives in the **chapter composition order** from `DESIGN_NOTES.md` Section 5. Chapter 2's minimal agent first, then layer additively. Resist single-shot implementation of the whole spec.
4. For each primitive: protocol file (interface), one reference implementation, one test file. Tests are the spec-conformance harness.
5. Build a `examples/` directory mirroring chapter labs as they're implemented (`examples/ch02_helixagent_v0_to_v1.py`, etc.).

## What to ask the author about

These are decisions the spec deliberately defers:

- Storage backend for the archive (SQLite for dev, Postgres for prod).
- Vector store adapters to ship in v1 (Turbopuffer + pgvector likely).
- Hook execution order policy (registration order vs explicit priority).
- MCP transport defaults (stdio is universal; HTTP and SSE first-class or opt-in?).
- The exact line between "scaffold" artifacts versus separate planner/monitor kinds.
- Whether Reflection is a Signal or a Search (open question in spec Section 13).

## What not to do

- Do not introduce new primitives beyond the ten defined in the spec without a spec amendment.
- Do not implement a UI, dashboard, model gateway, or workflow orchestrator in this repo.
- Do not bind to a single LLM provider in core code. Provider-specific code lives in adapters only.
- Do not let the agent loop accumulate branches. Capabilities go in hooks.
- Do not skip parent pointers on artifact mutation. Even quick experiments respect lineage.
- Do not generate prose with em-dashes or paragraphs longer than 3 sentences.

## Reference URLs

- LiteLLM docs: https://docs.litellm.ai
- Pydantic AI (not adopted but architecturally relevant): https://ai.pydantic.dev
- Instructor: https://python.useinstructor.com
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- OpenTelemetry Python: https://opentelemetry.io/docs/languages/python/

---

*This file lives at the repo root and is read by Claude Code on every session. Keep it short. Update only when non-negotiables or the base stack change.*
