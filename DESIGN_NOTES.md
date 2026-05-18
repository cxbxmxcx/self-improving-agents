# Design Decisions and Reasoning

Companion to SPEC.md. Captures the reasoning behind the architecture so future Claude Code sessions (and the author returning months later) understand *why* the design looks the way it does, not just *what* it is.

---

## 1. The framework question

The book's companion repository deliberately does not adopt a single agent framework as its spine. The reasoning, drawn from comparing the major 2026 options:

**LangGraph rejected for the spine.** LangGraph is the production winner in 2026 (34.5M monthly downloads, deployed at Klarna, Uber, Cisco, LinkedIn, BlackRock, JPMorgan), and its graph-based stateful workflows with checkpointing are a genuinely good fit for cyclic agent patterns. It was rejected because its strength (production-grade stateful workflows with built-in cycles and time-travel debugging) becomes pedagogical weight when the book's goal is to teach readers how to *design* improvement layers rather than *use* a framework's existing primitives. Adopting LangGraph would make Chapter 1 feel like a LangGraph tutorial.

**Pydantic AI considered and partially adopted.** Pydantic AI's type safety and small surface area make it strong for pedagogy. It was not adopted as the spine because (a) the book's thesis demands that readers see the agent loop as their own code, not as configuration of someone else's `Agent` class, and (b) lock-in concerns exist even with Pydantic AI's model-agnosticism because the `Agent` abstraction itself is the lock-in.

**OpenAI Agents SDK rejected.** The author's previous book (AI Agents in Action) used OpenAI Agents SDK. Repeating that choice would (a) limit the book's audience to OpenAI customers, (b) miss the chance to teach framework-independent agent thinking, and (c) feel architecturally identical to the previous book.

**Claude Agent SDK rejected for the spine but covered.** Strong production heritage (powers Claude Code) and Anthropic-native, but Claude-only lock-in repeats the OpenAI SDK problem from the previous book. The Claude Agent SDK's hooks system (PreToolUse, PostToolUse, etc.) directly inspired the Hook System in Section 6 of the spec. The SDK is worth a comparative chapter or sidebar.

**The decision: primitives-first, no framework spine.** Build directly on LiteLLM (model-agnostic LLM client), Pydantic v2 (typed inputs and outputs), and Instructor (for typed output with retry-on-validation). The agent loop is reader-written code, roughly 50-100 lines, that the reader sees once in Chapter 2 and understands completely. Every later chapter adds capabilities as hooks, signals, search methods, or artifacts, never as new framework abstractions.

This positions the book as the serious practitioner-author response to the 2026 framework wars rather than as a tutorial for one vendor's tooling.

---

## 2. The base stack

**LiteLLM** is the model-agnostic LLM client. It handles tool calling as a unified pass-through across all providers (OpenAI, Anthropic, Gemini, Bedrock, Ollama). The response shape is normalized so reader code looks the same regardless of provider.

**Pydantic v2** is the type system. Agent inputs, outputs, and state schemas are Pydantic models. Tool argument schemas are auto-derived from Python type hints via `create_model`. Most modern Python developers already know Pydantic, so the book teaches transferable engineering skills rather than framework-specific concepts.

**Instructor** wraps LiteLLM for typed Pydantic outputs with automatic retry on validation failure. Instructor is introduced in Chapter 2 or 3 as "the first improvement layer: validate, repair, retry," which doubles as a small self-improvement vignette before the book gets to the larger improvement layers.

**MCP Python SDK** handles MCP tool integration. Local Python tools (decorated with `@tool`) and MCP tools (loaded from `MCPToolset`) are normalized to the same Tool primitive inside the agent. The agent loop never branches on tool origin.

**OpenTelemetry** is the observability bus's wire format. The framework emits OTel spans and a typed event schema; readers plug in Phoenix, Langfuse, Logfire, LangSmith, Arize, or Braintrust as consumers. No bundled dashboard.

---

## 3. Pedagogical decisions

**The framework is its own repo, not the book.** The companion repository implements the spec completely. The book references the repo the way *Designing Data-Intensive Applications* references real systems: the architecture is what's taught, the code is where the curious go to verify. Manning MEAP readers get a working repo they can run; the book stays focused on the why and the what.

**Code listings are short and conceptual.** Typical listing: 5-20 lines that show the *shape* of a primitive or its *usage*, not its implementation. The reader sees `agent.search(prompt, signal=PairwiseJudge(), search=SPO())` in the book; they read the SPO implementation in the repo if they want to.

**Chapter 1 has minimal code.** Maybe one or two short listings showing the shape of the primitives. The listings exist to make the concepts concrete enough that the reader recognizes them in later chapters. Roughly 15 lines, not 150.

**Chapter 2 is where code earns real estate.** HelixAgent v0 to v1, RAG plus ReAct, SPO and GEPA against the routing and retrieval prompts. The reader sees Artifact, Signal, and Search primitives instantiated for the first time with real content.

**Chapters 3-7 reuse primitives aimed at different artifacts.** Same Signal protocol, same Search protocol, different `Artifact.kind`. This is the architectural payoff the spec is designed to produce: the search-by-signal grid is executable, not just illustrative.

**Chapters 8-11 are mostly orchestration and prose.** Tournament eval, HITL gates, multi-agent decision rubric, drift detection. Code here is short integration listings.

**Diagrams do the architectural work.** Budget of 60-80 diagrams. The search-by-signal grid figure (2D matrix with artifacts on one axis and search methods on the other, cells filled by canonical exemplars) is the book's intellectual map and will be referenced repeatedly.

---

## 4. The ten primitives at a glance

The spec defines ten primitives. They compose into the entire book. No new primitives are introduced after Chapter 1.

1. **Artifact** — anything mutable under search, versioned with parent pointers, content-addressed.
2. **Trajectory** — structured record of one agent run, replayable, the unit memory/eval/reflection learn from.
3. **Signal** — anything that returns a GapMeasurement; absorbs ground-truth, LLM-as-Judge, ContrastiveJudge, PRM, reflection, formal proof.
4. **Search** — proposes-and-selects variants guided by a signal; absorbs hill-climb, SPO, GEPA, MIPROv2, AFlow, DGM, MemRL, HyperAgents.
5. **Archive** — Pareto-aware store of historical variants with quality-diversity selection.
6. **Agent Loop + Hooks** — fixed canonical loop with nine hook points; never changes after Chapter 2.
7. **Memory** — four tiers (working, episodic, semantic, procedural) behind one contract; entries are artifacts.
8. **Eval** — tournament-capable, drift-aware; rubrics are artifacts that themselves decay.
9. **HITL** — approval gate, structured proposal with behavior diffs, four-tier ladder.
10. **Observability Bus** — OpenTelemetry events; drift detection, cost dashboards, runbooks are consumers.

---

## 5. The chapter composition map

How the primitives compose into each HelixAgent version:

- **v0 (Ch 2)** — Minimal agent: system prompt as Artifact, tools, working memory. Pure agent loop, no improvement.
- **v1 (Ch 2)** — Add Signal (LLM-as-Judge pairwise) + Search (SPO) aimed at system prompt. First closed self-improvement loop.
- **v2 (Ch 3)** — Add episodic + semantic memory tiers. PRE_MODEL hook reads memory, SESSION_END hook writes trajectory.
- **v3 (Ch 4)** — Add MemRL as a Search aimed at episodic entries. Add ExpeL insight extraction. Add 3 AM offline consolidation.
- **v4 (Ch 5)** — Add Planner/Monitor/Reflector/TSM as three coordinated hooks. Reflection is a Signal applied at SESSION_END.
- **v5 (Ch 6)** — Aim existing searches at skill and tool_description artifacts. Add HITL approval gate on archive commit.
- **(Ch 7)** — Thought experiments: what each frontier system (DGM, AlphaEvolve, HyperAgents) would compose differently. Primitives unchanged.
- **v6-v8 (Ch 8-11)** — Eval subsystem replaces ad-hoc signals. HITL ladder configured per artifact kind. Multi-agent is composition (sub-agents are tools). Drift detection consumes the bus.

By Chapter 11 the framework has accumulated hooks, signal implementations, and consumers, all within the ten primitives. The agent loop is the same eight-line skeleton it was in Chapter 2.

---

## 6. Open architectural questions

Items where the spec is provisional and a manuscript decision is pending:

**Reflection as Signal versus Search.** The spec puts Reflection in the Signal family (Section 3.3) because it makes GEPA-uses-reflection clean. An alternative is to treat Reflexion as a Search that consumes its own Signal output. Chapter 5 will validate or revise.

**Eval as primitive versus composition.** Section 8 makes Eval a primitive, but it could be expressed as a Signal-plus-Archive composition with a calibration Signal layered on. Pedagogically cleaner as primitive; architecturally cleaner as composition. Decision pending Chapter 8 drafting.

**Multi-agent composition shape.** Section 11.8 treats sub-agents as tools. Chapter 10's Anthropic-versus-Cognition rubric may demand a richer composition primitive (orchestrator-worker as a distinct shape, group-chat as another). Open until Chapter 10 is drafted.

**The artifact kind for planner/monitor scaffolds.** Currently two kinds (`planner`, `monitor`). May be cleaner to unify under a single `scaffold` kind with a sub-type. Deferred to Chapter 5 drafting.

**Code as an artifact kind.** Section 1.2 lists `code` as a recognized kind. Execution sandbox and diff representation for code artifacts are not specified. Chapter 7 frontier territory and deliberately under-specified.

**Trajectory storage cost.** Trajectories are large. Long-running deployments accumulate trajectories faster than archives. The spec assumes a TTL plus sampling policy; details depend on Chapter 8's continuous-eval sampling strategy, which is still in design.

**Skills as a separate subsystem.** Currently folded under procedural memory. May need its own section by Chapter 6.

**MCP as a protocol layer.** Folded under tools currently. May need promoting to its own section given MCP's centrality in 2026.

---

## 7. House style for generated content

Any prose, documentation, or commentary generated for this project follows these rules:

- **Paragraphs are 3 sentences.** Hard rule for prose deliverables. Break longer paragraphs at natural conceptual seams.
- **No em-dashes.** Use commas, semicolons, or colons instead.
- **Markdown only for documents.** No .docx unless explicitly requested.
- **NPG color palette where colors apply:** vermillion (#E64B35), slate blue (#3C5488), ocean teal (#00A087), soft cyan (#4DBBD5), muted salmon (#F39B7F). Nature Editorial infographic aesthetic.
- **Code listings are short.** 5-20 lines for in-book listings. Repository code can be longer but stays readable.
- **Mermaid for architectural diagrams.** All diagram sources in Mermaid.

---

## 8. Non-goals (what the framework deliberately does not provide)

- A UI. Reviewers, dashboards, operator interfaces are not in scope.
- A model gateway. LLM routing, fallback, cost optimization handled by LiteLLM or the reader's existing gateway.
- A vector database. Memory tiers target existing vector stores (Turbopuffer, pgvector, Pinecone) through adapters.
- A workflow orchestrator. The agent loop is the loop. Multi-step workflows are agents composed of tools.
- Fine-tuning, RLHF, or weight updates. Out of scope by the book's thesis.

---

## 9. Decisions still ahead (for the implementation phase)

When implementing the spec, expect to make calls on:

- **Storage backend for archive.** SQLite for local development, Postgres for production. Migration path between them.
- **Vector database adapter set.** Which vector stores ship with adapters in v1. Likely Turbopuffer (author preference) and pgvector (open-source baseline).
- **Trajectory serialization format.** JSON Lines is the obvious default. Consider Arrow or Parquet for large archives.
- **Hook execution order policy.** Registration order, priority levels, or both?
- **Search budget enforcement.** Hard cutoff versus soft warning when budget exceeded.
- **MCP transport defaults.** Stdio is universal; should HTTP and SSE be first-class or opt-in?
- **Calibration set management.** How readers contribute their own calibration cases versus framework-provided defaults.
- **Multi-tenancy in shared deployments.** How scope (per-user, per-org, global) interacts with archive persistence.

Each is a Chapter-N implementation decision, not a Chapter 1 spec decision.
