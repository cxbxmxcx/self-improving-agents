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

---

## 10. Promotion: offline gates and online auto-apply

Chapter 1 introduces the online versus offline distinction as two cadences of the same loop, and the safety rule attached to it: online loops touch only L1 and L2 artifacts. The platform has to enforce that rule structurally, because if it does not, the chapter's framing contradicts what the code actually does. This section captures the design that makes the platform honest.

**The core split: candidate ranking versus live champion.** Before this design, "the current best" was a single concept: `archive.best()` returned the highest-scoring measured artifact, and the agent used it. After this design, those are two distinct things. The *candidate ranking* is a property of the archive and updates every round; the *live champion* is a property of the deployment and updates only when something calls `archive.promote()`. In online mode they stay in lockstep because the Improver auto-promotes; in offline mode they can drift indefinitely while a human reviews.

**Promotion is the act of changing what `archive.live_champion(id)` returns.** It carries three things, because audit and rollback depend on all three: which candidate (artifact id and version), who promoted it (a user id, or `improver:<id>` when automatic), and why (the round result that justified it, or a human note). A rollback is "promote a previous version with reason='rollback'." Same primitive, no special case.

**Mode lives on `ImproverPolicy`, but the policy is executed by a hook.** The Improver carries an `ImproverMode` enum on its policy (OFFLINE default, ONLINE opt-in). At the end of each round, if the candidate beat the reference, the Improver emits a `candidate_wins` event carrying the candidate, the reference, the round result, and the mode. A single global hook handler listens for this event and switches on mode: ONLINE calls `archive.promote(approver=f"improver:{id}", reason=round_result)`; OFFLINE does nothing. Users who want custom promotion (confidence thresholds, Pareto dominance, two-judge agreement) register their own handler and set `policy.auto_promote=False` to disable the default.

This is Option B from the design discussion (hook-based) rather than Option A (Improver-internal). It costs slightly more code than putting the promote call inside the Improver, but it gives users a clean extension point that does not require subclassing, and it makes promotion observable as a first-class event rather than as a side effect of round completion.

**The hook fires after the archive has recorded the candidate.** Order within a round: search produces candidate, signal scores it, archive records candidate and measurement (always), then if the candidate won, the bus emits `candidate_wins`. Hook handlers run on that event. This means offline candidates are visible in the dashboard the moment they are produced, even if no human ever promotes them. That matches the chapter's framing of the deploy gate as a *review* checkpoint, not a *suppression* checkpoint: the gate is supposed to let humans inspect everything the loop has tried, not hide non-promoted work.

**Two separate events.** `candidate_wins` says "a candidate just beat the reference in a round." `promoted` says "the live champion just changed." Online winners produce both events back to back. Offline winners produce `candidate_wins` only; the `promoted` event fires later when a human clicks Promote. The split lets observability subscribers distinguish "the loop is finding improvements" from "improvements are reaching users," and it makes a "stalled gate" alert trivial to build later: count `candidate_wins` without a matching `promoted` within N hours.

**The live champion is global per artifact id, not scoped per agent.** Two agents sharing a prompt id share a live champion. This matches the chapter's framing of artifacts as shared infrastructure ("a single-agent definition runs as many concurrent instances ... drawing on the same prompts, memory store, and tools"). The trade-off is real: a promotion in one agent's improver affects every agent reading that artifact. The mitigation is naming: agent specs should use distinct prompt ids when they want independent improvement (`prompt.helpdesk.system` versus `prompt.researcher.system`), and the platform leaves that decision to the spec author rather than imposing per-agent scoping.

**In-flight requests survive mid-round promotions.** The Agent reads `live_champion(id)` once at request start. A request running with v9 finishes with v9 even if v10 was promoted at second three. This is correct and documented as an invariant. It means `live_champion()` reads need to be cheap (the implementation caches the result and invalidates on `promoted` events), and it means an online loop's effective change rate is bounded by request duration, not by round duration.

**The online safety check is structural, not advisory.** When `ImproverPolicy.mode == ONLINE`, the Improver's constructor inspects the target artifact's layer (L1 prompt, L2 memory, L3 metacognition, L4 code-and-tools) and raises if it sees L3 or L4. This requires adding a `layer` property to ArtifactKind, computed from a small map. The Improver does not warn or log; it refuses to construct. The chapter's rule ("online loops can only safely change prompts and memory") is enforced by the type system rather than by documentation.

**What the UI surfaces.** The chat UI's improver card shows a mode badge (offline shield, online lightning bolt) and, when an offline improver has a winning candidate not yet live, a single "Promote v{N} → live" button. Online cards never show the button because the candidate is already live. The dashboard's champion panel shows two facts per artifact family: "live champion" (what users see) and "best candidate" (highest score in archive); when they differ, the dashboard highlights the gap and offers a per-row promote button so a reviewer can act from the lineage view as well.

---

## 11. Multi-artifact agents: one improver per artifact

Real agents have multiple artifacts under simultaneous improvement: a system prompt at L1, an episodic memory layout at L2, a tool implementation at L4, a tool description back at L1, possibly a guardrail at L4. The architectural question this raises: do we run one improver per artifact, or one improver per agent that coordinates over all of them?

**The decision: one improver per artifact, sharing a factory (SPEC §16.1).** Four alternatives were considered:

1. **Parallel per-artifact improvers** — what we now have.
2. **Single agent-level improver with sub-searches** — one improver mutates all artifacts per round and measures the bundle.
3. **Layered sequential search** — improve L1 to convergence, then L2, then L4.
4. **Per-artifact improvers with an agent-level regression checker** — improvers run locally, a separate watcher catches combination drift.

Option 1 won for three reasons. **(a) The per-artifact improver is teachable as a unit.** "One artifact, one improver, one signal, one search" is the primitive readers can hold in their head; collapsing to a coordinator trades clarity for joint optimality, and the book has to introduce concepts in order. **(b) Most artifacts are weakly coupled in practice.** A memory entry's value rarely depends on the exact wording of the system prompt; a prompt's quality rarely depends on tool internals. Joint search is solving a problem that mostly is not there. **(c) When local-good-global-bad does happen, the next round catches it.** Each improver's signal measures the *whole agent*, not the bare artifact. The score recorded against a prompt candidate is the score of that prompt running with the current live memory and the current live tool code; combination drift surfaces in the next round's measurement, not in a separate guardrail.

**Why Option 4 (AgentRegression) was considered and dropped.** An earlier draft of this design added a separate watcher that re-evaluated the full live combination on every promotion and rolled back regressions. It got dropped because each improver's signal already evaluates the full agent; a separate regression check would double the eval cost without adding new information. The promotion log plus measurement attribution by signal (§5.2.1) gives the same audit trail.

**Why the factory resolves dependencies, not the improver.** When an improver tests a candidate, the factory call needs the full artifact bundle. Two options for where the missing artifacts come from: the improver looks them up before calling the factory, or the factory looks them up internally. The factory won because (a) the factory is the single source of truth for what artifacts compose the agent; putting the dependency list on each improver duplicates that knowledge and the two can drift apart, (b) production deployment calls the same factory, so the same code path serves training and serving, (c) the improver's signature stays minimal — `target_artifact_id` plus the standard pieces, no `dependencies=[...]` parameter.

**The eventual-consistency property.** Each improver round measures its candidate against the artifacts that are live at the moment the round runs. If another improver promotes between rounds, subsequent measurements pick up the new live combination automatically. There is no synchronization barrier, no coordinated rollback, no "freeze all improvers while one is running" protocol. The framework trusts the next round to correct any drift; the dashboard surfaces score trends so operators can see if drift is accumulating.

**MultiArtifactImprover is mentioned as a future extension point, not built.** When per-artifact improvement is insufficient (genuinely strongly-coupled artifacts), a subclass that mutates multiple artifacts per round and measures the bundle is the right escalation. The framework hooks for this are the existing Improver class; the chapter that needs it (Ch 8+ if ever) can introduce it without changing the foundation.

---

## 12. Signal extensions: identity, thresholds, and metric signals

The signal layer had three gaps relative to the chapters that need to come next. All three were filled additively (SPEC §3) without breaking existing signal implementations.

**Gap 1: signals had no identity.** A measurement in the archive carried `score`, `preference`, `feedback`, and a `rubric_id`, but no field naming the signal that produced it. When a new signal is introduced, prior measurements become silently incomparable. The fix is `signal_id` + `signal_version` on the Signal protocol and on every GapMeasurement, persisted to the archive. Two CompositeSignal instances with different weights are *different signals* and carry different ids; the archive's `measurements_for_signal()` read path lets tooling decide what to do with prior incomparable measurements (re-measure, ignore, or trend separately).

**Gap 2: composite signals had no normalization step.** `CompositeSignal._aggregate_score` does weighted mean on raw scores, which is fine when every child returns a value in `[0, 1]`. A judge returning 0.7 weighted against a token-count signal returning 4200 would be dominated by raw magnitude. The fix is a `SignalThreshold` mixin (baseline + threshold + direction + normalizer) that any non-`[0, 1]` signal embeds to satisfy the "normalized score in `[0, 1]`" contract. Existing judge signals don't embed it; their scores are already normalized.

**Gap 3: no metric signal family.** Token counts, latency, tool-call counts, model-call counts are first-class inputs to improvement decisions. A prompt that improves judge score by 0.05 but costs 3× the tokens may not be the candidate to promote. The fix is `SignalKind.METRIC` and a `MetricSignal` class that reads the relevant counter from a Trajectory, normalizes via `SignalThreshold`, and emits a GapMeasurement with `raw_value` and `score` filled. CompositeSignal can now mix a judge and a metric signal with explicit weights, and the dashboard can surface both the observed quantity and its normalized contribution.

**Why thresholds live on the signal, not on the improver.** An earlier draft put trigger thresholds on `ImproverPolicy` alongside `promote_threshold_win_rate`. They got moved to the signal so that (a) composite signals can report "child N triggered even though the aggregate did not" via `metadata["component_triggers"]`, (b) the `triggered` flag on GapMeasurement is portable across improvers and dashboards without re-deriving the threshold, (c) the promotion-threshold concept on ImproverPolicy stays orthogonal to the gap-trigger concept on Signal. The two thresholds answer different questions: "did this candidate's signal fire?" versus "did this candidate beat the reference enough to promote?"

---

## 13. Agent-side artifact wrappers: Guardrail and ToolFromArtifact

The framework supports two artifact kinds that bolt onto the agent's lifecycle in structured ways: guardrails (CODE at L4) and tools (CODE at L4 paired with TOOL_DESCRIPTION at L1). Both are wired via convenience wrappers — `Guardrail` and `ToolFromArtifact` — that turn artifacts into typed Agent constructor parameters. Neither is a new primitive.

**Why guardrails are CODE (L4), not a new artifact kind.** An earlier draft introduced `ArtifactKind.GUARDRAIL` as an L1 kind on the theory that guardrail "rules" were prompt-like. That was wrong: most production guardrails (PII regex, schema validation, length caps, refusal lists) are deterministic Python, not LLM-driven judgment. They are exactly what `ArtifactKind.CODE` already represents. Grounding and critic agents — which *are* LLM-driven and *do* take prompts — are not guardrails; they are agents in their own right, with their own L1 prompt artifacts and their own improvers. The multi-agent chapter demonstrates them as a composition pattern (an answer agent plus a critic agent, each independently improved) without needing a new artifact kind.

**Why guardrails are typed Agent constructor parameters, not generic hooks.** Hooks already exist (SPEC §6.2) and a guardrail could be implemented as a hook that loads a CODE artifact at construction. Three reasons to elevate guardrails to a typed parameter instead: (a) the typed surface advertises the agent's safety posture — reading the constructor tells you what guardrails are attached, (b) the framework can validate at construction that the artifact's compiled function matches the input/output guardrail contract, (c) production deployment uses the same wrapper, so the lineage of which guardrail version was active for which request is recorded automatically in `Trajectory.artifacts_used`.

**Why refusal is the only short-circuit behavior.** When an output guardrail fails, two patterns exist in the wild: refuse, or loop back and have the agent retry with the guardrail's feedback as context. The framework supports refusal only. The retry pattern is structurally a critic agent — a second agent that judges and rewrites — and belongs to the multi-agent chapter, not to the guardrail wrapper. Conflating the two would make the guardrail's contract ambiguous (does `allow=False` mean "stop" or "try again"?) and would make signal interpretation hard ("did the original answer score 0.7 because it was 0.7 quality, or because the critic rewrote it to 0.7?"). The clean split: guardrails stop, critics rewrite.

**Why tools sourced from artifacts need two artifacts, not one.** A tool's implementation is L4 code; its LLM-facing description is L1 text. They have different improvement economics: the implementation needs sandboxed code evolution with a test suite, the description can be mutated by SPO with a single LLM call. Bundling them into one artifact would force the description's improver to wait on the implementation's CI cadence, or force the implementation's improver to also run text mutation. The dual-artifact pattern lets each side improve independently while `ToolFromArtifact` keeps the user-facing wiring as one object.
