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

**Chapter 2 is where code earns real estate.** HelixAgent v0 to v1, RAG plus ReAct, SPO and a hill-climb against the system prompt, with GEPA and DGM following in Chapter 4. The reader sees Artifact, Signal, and Search primitives instantiated for the first time with real content.

**Chapters 3 to 8 reuse primitives aimed at different artifacts.** Same Signal protocol, same Search protocol, different `Artifact.kind`. This is the architectural payoff the spec is designed to produce: the search-by-signal grid is executable, not just illustrative.

**Chapters 9 through 12 are mostly orchestration and prose.** Tournament eval, HITL gates, multi-agent decision rubric, drift detection. Code here is short integration listings.

**Diagrams do the architectural work.** Budget of 60-80 diagrams. The search-by-signal grid figure (2D matrix with artifacts on one axis and search methods on the other, cells filled by canonical exemplars) is the book's intellectual map and will be referenced repeatedly.

---

## 4. The ten primitives at a glance

The spec defines ten primitives. They compose into the entire book. No new primitives are introduced after Chapter 1.

1. **Artifact**: anything mutable under search, versioned with parent pointers, content-addressed.
2. **Trajectory**: structured record of one agent run, replayable, the unit memory/eval/reflection learn from.
3. **Signal**: anything that returns a GapMeasurement; absorbs ground-truth, LLM-as-Judge, ContrastiveJudge, PRM, reflection, formal proof.
4. **Search**: proposes-and-selects variants guided by a signal; absorbs hill-climb, SPO, GEPA, MIPROv2, AFlow, DGM, MemRL, HyperAgents.
5. **Archive**: Pareto-aware store of historical variants with quality-diversity selection.
6. **Agent Loop + Hooks**: fixed canonical loop with nine hook points; never changes after Chapter 2.
7. **Memory**: four tiers (working, episodic, semantic, procedural) behind one contract; entries are artifacts.
8. **Eval**: tournament-capable, drift-aware; rubrics are artifacts that themselves decay.
9. **HITL**: approval gate, structured proposal with behavior diffs, four-tier ladder.
10. **Observability Bus**: OpenTelemetry events; drift detection, cost dashboards, runbooks are consumers.

---

## 5. The chapter composition map

How the primitives compose into each HelixAgent version:

- **v0 (Ch 2)**: Minimal agent: system prompt as Artifact, tools, working memory. Pure agent loop, no improvement.
- **v1 (Ch 2)**: Add Signal (LLM-as-Judge pairwise) + Search (SPO, hill-climb) aimed at the system prompt; the archive and offline promote. First closed self-improvement loop.
- **v1 (Ch 3, the Gödel Machine)**: No new hooks or searches; the chapter builds out the signal side: deterministic checkers, LLM judges, the proof gate (`proven` on GapMeasurement), and Reflection as a Signal filling `feedback`. The version label does not advance because the agent's artifacts do not change.
- **v2 (Ch 4, the Darwin Gödel Machine)**: Add population search (GEPA, DGM, StrategyChain) sharing one archive on a deterministic task-success signal. The Observatory arrives as a bus consumer outside `helix/`.
- **v3 (Ch 5, memory)**: Install memory as a harness component with PRE_MODEL and SESSION_END hook wrappers, and retain entries as `MEMORY_ENTRY` artifacts. Add `MemoryLearner` for fixed one-step value-based learning, `SignalKind.UTILITY`, memory operators, `OnlineMemoryImprover`, group-relative lesson selection, and consolidation. `MemRLSearch` remains a Search adapter; learning changes stored values while the configured application mechanism and learning algorithm stay fixed.
- **v4 (Ch 6, metacognition)**: Add planner, monitor, reflector, and task-state hook wrappers bound in a `METACOGNITION` composite. Add the behavior-diff signal, the dual-process router at PRE_MODEL, and `ComposedSearch` over the composite.
- **v5 (Ch 7, the harness)**: A coding agent with skill files (`helix/skills/`), `TextDescriptionTool`, a swappable sandboxed `ToolFromArtifact`, a `CodeMutator` with DGM aimed at code, and the `HARNESS` composite. Add the minimal HITL gate (Proposal, Decision, behavior diffs, rollback) as a `candidate_wins` subscriber.
- **(Ch 8, HyperAgents)**: Add `SearchKind.EDITABLE_META` with the mutator prompt as a TEXT artifact and a read-only policy at the meta level. Primitives unchanged.
- **v6 (Ch 9, evaluation)**: Eval subsystem replaces ad-hoc signals: tournament eval, calibration drift, continuous eval, judge bias.
- **v7 (Ch 10, humans)**: HITL ladder configured per artifact kind, confidence routing, inoculation review, the drift queue, live feedback signals, and HITL-gated online prompt improvement.
- **(Ch 11, multi-agent)**: Multi-agent is composition; sub-agents are tools. No new primitives.
- **v8 (Ch 12, shipping)**: Drift detection consumes the bus; rollout controls and the reward-hacking runbook are consumers too.

By Chapter 12 the framework has accumulated hooks, signal implementations, and consumers, all within the ten primitives. The agent loop is the same eight-line skeleton it was in Chapter 2.

---

## 6. Open architectural questions

Items where the spec is provisional and a decision is pending:

**Reflection as Signal versus Search (resolved, v0.3).** The spec puts Reflection in the Signal family (Section 3.3) because it makes GEPA-uses-reflection clean. The alternative, Reflexion as a Search consuming its own Signal output, was rejected in the Chapter 6 plan: Reflection stays a Signal, and its only route to changed behavior is a memory entry read at PRE_MODEL. See §17.

**Eval as primitive versus composition.** Section 8 makes Eval a primitive, but it could be expressed as a Signal-plus-Archive composition with a calibration Signal layered on. Pedagogically cleaner as primitive; architecturally cleaner as composition. Decision pending Chapter 9 drafting.

**Multi-agent composition shape.** Section 11.9 treats sub-agents as tools. Chapter 11's Anthropic-versus-Cognition rubric may demand a richer composition primitive (orchestrator-worker as a distinct shape, group-chat as another). Open until Chapter 11 is drafted.

**The artifact kind for planner/monitor scaffolds (resolved, SPEC v0.2).** Planner and monitor are `text` subtypes at L1, not kinds of their own; the L3 risk lives on the `metacognition` composite that binds them (§15). Chapter 6 builds on that shape.

**Code as an artifact kind.** Section 1.2 lists `code` as a recognized kind. The execution sandbox and the unified-diff mutation representation are specified by the Chapter 7 plan (a subprocess sandbox with a timeout, diffs stored in artifact metadata, no promotion without a passing test signal). Chapter 7 is where code comes under search; the earlier "Chapters 11/12" deferral is gone.

**Trajectory storage cost.** Trajectories are large. Long-running deployments accumulate trajectories faster than archives. The spec assumes a TTL plus sampling policy; details depend on Chapter 9's continuous-eval sampling strategy, which is still in design.

**Skills as a separate subsystem (decided).** Skills stay backed by the procedural memory tier, but they get their own package: `helix/skills/` (loader, SKILL.md reader, skill-to-prompt and skill-as-tool bridges) lands in Chapter 7. Until then `Subtype.SKILL` exists and nothing in the loop reads it.

**MCP as a protocol layer (decided for now).** MCP stays folded under tools; local tools and MCP tools normalize to the same Tool primitive. It is listed in the base stack but absent from the code and requirements today, and it enters only if the Chapter 7 harness grows an MCP surface.

---

## 7. Style for repository content

Any prose, documentation, or commentary generated for this project follows these rules:

- **Paragraphs are 3 sentences.** Hard rule for prose deliverables. Break longer paragraphs at natural conceptual seams.
- **No em-dashes.** Use commas, semicolons, or colons instead.
- **Markdown for documents.**
- **NPG color palette where colors apply:** vermillion (#E64B35), slate blue (#3C5488), ocean teal (#00A087), soft cyan (#4DBBD5), muted salmon (#F39B7F). Nature Editorial infographic aesthetic.
- **Code stays readable.** Chapter listings are short, and the code they excerpt stays readable.

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

1. **Parallel per-artifact improvers**: what we now have.
2. **Single agent-level improver with sub-searches**: one improver mutates all artifacts per round and measures the bundle.
3. **Layered sequential search**: improve L1 to convergence, then L2, then L4.
4. **Per-artifact improvers with an agent-level regression checker**: improvers run locally, a separate watcher catches combination drift.

Option 1 won for three reasons. **(a) The per-artifact improver is teachable as a unit.** "One artifact, one improver, one signal, one search" is the primitive readers can hold in their head; collapsing to a coordinator trades clarity for joint optimality, and the book has to introduce concepts in order. **(b) Most artifacts are weakly coupled in practice.** A memory entry's value rarely depends on the exact wording of the system prompt; a prompt's quality rarely depends on tool internals. Joint search is solving a problem that mostly is not there. **(c) When local-good-global-bad does happen, the next round catches it.** Each improver's signal measures the *whole agent*, not the bare artifact. The score recorded against a prompt candidate is the score of that prompt running with the current live memory and the current live tool code; combination drift surfaces in the next round's measurement, not in a separate guardrail.

**Why Option 4 (AgentRegression) was considered and dropped.** An earlier draft of this design added a separate watcher that re-evaluated the full live combination on every promotion and rolled back regressions. It got dropped because each improver's signal already evaluates the full agent; a separate regression check would double the eval cost without adding new information. The promotion log plus measurement attribution by signal (§5.2.1) gives the same audit trail.

**Why the factory resolves dependencies, not the improver.** When an improver tests a candidate, the factory call needs the full artifact bundle. Two options for where the missing artifacts come from: the improver looks them up before calling the factory, or the factory looks them up internally. The factory won because (a) the factory is the single source of truth for what artifacts compose the agent; putting the dependency list on each improver duplicates that knowledge and the two can drift apart, (b) production deployment calls the same factory, so the same code path serves training and serving, (c) the improver's signature stays minimal: `target_artifact_id` plus the standard pieces, no `dependencies=[...]` parameter.

**The eventual-consistency property.** Each improver round measures its candidate against the artifacts that are live at the moment the round runs. If another improver promotes between rounds, subsequent measurements pick up the new live combination automatically. There is no synchronization barrier, no coordinated rollback, no "freeze all improvers while one is running" protocol. The framework trusts the next round to correct any drift; the dashboard surfaces score trends so operators can see if drift is accumulating.

**MultiArtifactImprover is mentioned as a future extension point, not built.** When per-artifact improvement is insufficient (genuinely strongly-coupled artifacts), a subclass that mutates multiple artifacts per round and measures the bundle is the right escalation. The framework hooks for this are the existing Improver class; the chapter that needs it (Ch 9 or later, if ever) can introduce it without changing the foundation.

---

## 12. Signal extensions: identity, thresholds, and metric signals

The signal layer had three gaps relative to the chapters that need to come next. All three were filled additively (SPEC §3) without breaking existing signal implementations.

**Gap 1: signals had no identity.** A measurement in the archive carried `score`, `preference`, `feedback`, and a `rubric_id`, but no field naming the signal that produced it. When a new signal is introduced, prior measurements become silently incomparable. The fix is `signal_id` + `signal_version` on the Signal protocol and on every GapMeasurement, persisted to the archive. Two CompositeSignal instances with different weights are *different signals* and carry different ids; the archive's `measurements_for_signal()` read path lets tooling decide what to do with prior incomparable measurements (re-measure, ignore, or trend separately).

**Gap 2: composite signals had no normalization step.** `CompositeSignal._aggregate_score` does weighted mean on raw scores, which is fine when every child returns a value in `[0, 1]`. A judge returning 0.7 weighted against a token-count signal returning 4200 would be dominated by raw magnitude. The fix is a `SignalThreshold` mixin (baseline + threshold + direction + normalizer) that any non-`[0, 1]` signal embeds to satisfy the "normalized score in `[0, 1]`" contract. Existing judge signals don't embed it; their scores are already normalized.

**Gap 3: no metric signal family.** Token counts, latency, tool-call counts, model-call counts are first-class inputs to improvement decisions. A prompt that improves judge score by 0.05 but costs 3× the tokens may not be the candidate to promote. The fix is `SignalKind.METRIC` and a `MetricSignal` class that reads the relevant counter from a Trajectory, normalizes via `SignalThreshold`, and emits a GapMeasurement with `raw_value` and `score` filled. CompositeSignal can now mix a judge and a metric signal with explicit weights, and the dashboard can surface both the observed quantity and its normalized contribution.

**Why thresholds live on the signal, not on the improver.** An earlier draft put trigger thresholds on `ImproverPolicy` alongside `promote_threshold_win_rate`. They got moved to the signal so that (a) composite signals can report "child N triggered even though the aggregate did not" via `metadata["component_triggers"]`, (b) the `triggered` flag on GapMeasurement is portable across improvers and dashboards without re-deriving the threshold, (c) the promotion-threshold concept on ImproverPolicy stays orthogonal to the gap-trigger concept on Signal. The two thresholds answer different questions: "did this candidate's signal fire?" versus "did this candidate beat the reference enough to promote?"

---

## 13. Agent-side artifact wrappers: Guardrail and ToolFromArtifact

The framework supports two artifact kinds that bolt onto the agent's lifecycle in structured ways: guardrails (CODE at L4) and tools (CODE at L4 paired with TOOL_DESCRIPTION at L1). Both are wired via convenience wrappers (`Guardrail` and `ToolFromArtifact`) that turn artifacts into typed Agent constructor parameters. Neither is a new primitive.

**Why guardrails are CODE (L4), not a new artifact kind.** An earlier draft introduced `ArtifactKind.GUARDRAIL` as an L1 kind on the theory that guardrail "rules" were prompt-like. That was wrong: most production guardrails (PII regex, schema validation, length caps, refusal lists) are deterministic Python, not LLM-driven judgment. They are exactly what `ArtifactKind.CODE` already represents. Grounding and critic agents, which *are* LLM-driven and *do* take prompts, are not guardrails; they are agents in their own right, with their own L1 prompt artifacts and their own improvers. The multi-agent chapter demonstrates them as a composition pattern (an answer agent plus a critic agent, each independently improved) without needing a new artifact kind.

**Why guardrails are typed Agent constructor parameters, not generic hooks.** Hooks already exist (SPEC §6.2) and a guardrail could be implemented as a hook that loads a CODE artifact at construction. Three reasons to elevate guardrails to a typed parameter instead: (a) the typed surface advertises the agent's safety posture: reading the constructor tells you what guardrails are attached, (b) the framework can validate at construction that the artifact's compiled function matches the input/output guardrail contract, (c) production deployment uses the same wrapper, so the lineage of which guardrail version was active for which request is recorded automatically in `Trajectory.artifacts_used`.

**Why refusal is the only short-circuit behavior.** When an output guardrail fails, two patterns exist in the wild: refuse, or loop back and have the agent retry with the guardrail's feedback as context. The framework supports refusal only. The retry pattern is structurally a critic agent, a second agent that judges and rewrites, and belongs to the multi-agent chapter, not to the guardrail wrapper. Conflating the two would make the guardrail's contract ambiguous (does `allow=False` mean "stop" or "try again"?) and would make signal interpretation hard ("did the original answer score 0.7 because it was 0.7 quality, or because the critic rewrote it to 0.7?"). The clean split: guardrails stop, critics rewrite.

**Why tools sourced from artifacts need two artifacts, not one.** A tool's implementation is L4 code; its LLM-facing description is L1 text. They have different improvement economics: the implementation needs sandboxed code evolution with a test suite, the description can be mutated by SPO with a single LLM call. Bundling them into one artifact would force the description's improver to wait on the implementation's CI cadence, or force the implementation's improver to also run text mutation. The dual-artifact pattern lets each side improve independently while `ToolFromArtifact` keeps the user-facing wiring as one object.

---

## 14. OfflineImprover vs OnlineImprover: two classes, not one

Earlier drafts had a single `Improver` class with an `ImproverMode` enum (OFFLINE / ONLINE) on its policy. That worked mechanically: an offline improver took a `FixedEvalSet`, an online improver took a `LiveTrafficSource`, and a hook chain dispatched on the mode to decide whether to auto-promote. But the configuration was hiding a real architectural difference. Author decision: split into two peer classes, `OfflineImprover` and `OnlineImprover`, sharing no base class.

**Why peer classes, not subclasses.** Two classes can satisfy the same protocols without sharing a base class, and that is the right shape when their lifecycles genuinely differ. OfflineImprover owns a driver loop and tests candidates against a labeled set; OnlineImprover has no loop and reacts to the agent's `SESSION_END` hook. A shared base class would either be empty (just hierarchy with no shared code) or it would force one of them into the other's shape. The framework already takes the peer-classes-share-protocols stance for Signal families (`PairwiseJudge` and `LiveTrajectoryJudge` are peers, not parent/child); the same logic applies here.

**Why rename, not deprecate.** The codebase is pre-release, not a stable release. Keeping `Improver` as a deprecated alias for `OfflineImprover` for one cycle would carry the cost of dual paths in every chapter script and in every test. Author decision: rename cleanly, no alias. Existing call sites get updated in the same change set. This breaks no production users because there are no production users yet.

**Why "improver" stays as a category noun.** The spec uses the word "improver" in prose to refer to either class. There is no `Improver` Protocol or base class; the word is plain English. This matches how the spec uses "signal family" as a category noun without requiring all signals to share a base class. Readers refer to "the improver" without the type system pretending there's a unifying type underneath.

**Why `Agent.with_artifacts` replaces the factory function.** Earlier drafts asked the user to write a `build_agent_with_artifacts(artifacts: dict, archive: Archive) -> Agent` function and pass it as a kwarg to the Improver. That worked but was noise: the user wrote a function whose body was mostly "construct the agent the same way every time, but with this one slot variable." Author decision: replace the factory contract with a `with_artifacts` method on Agent itself. The user writes one Agent definition; the improver introspects everything it needs.

This is the same trade-off the spec already accepted for Signal: `signal.measure(candidate, trajectory, ...)` is a method on Signal, not a free function the user passes in. Putting the clone logic on Agent is consistent.

**Why no `agent_factory=` parameter on Search methods either.** GEPA used to take `agent_factory=` so it could iterate over a population of candidates. The same `Agent.with_artifacts` shape applies: GEPA takes `agent=` and calls `agent.with_artifacts({target_id: candidate})` internally. The user-facing surface across Improver and Search becomes uniform.

**Why OnlineImprover has no eval_source.** OfflineImprover takes an `EvalSource` because it needs questions to test candidates against. OnlineImprover does not, because the agent's live trajectories *are* the source of measurement. Adding an eval_source parameter to OnlineImprover would be confusing (what would it mean?) and would invite a hybrid pattern (online improver tests candidates against a held-out eval set) that is achievable today by writing a small subclass but is not the primary contract.

**Why `agent.attach_improver` is online-only meaningful.** OfflineImprover builds its own agent clones; it does not need the agent to be running or attached. OnlineImprover subscribes to the agent's `SESSION_END` hook, so it only works after attachment. The chapter scripts call `attach_improver` for OfflineImprover too, but that's a teaching convenience (it lets the dashboard show which improvers are watching which agent); the call is structurally a no-op for offline improvement.

---

## 15. The TEXT kind and the kind-versus-layer split (SPEC v0.2)

Earlier drafts had a per-role artifact kind enum: `prompt`, `skill`, `tool_description`, `rubric`, `planner`, `monitor`, plus `memory_entry` and `code`. The roles were proliferating, and they hid a fact the search layer kept rediscovering: a prompt, a tool description, a skill, and a rubric are all natural language the LLM reads, and they are all mutated by the same text search methods (SPO, GEPA, reflective mutation). SPEC §4.3 already said as much in prose ("SPO works on prompts, on memory entries, on skill files, on rubrics"). Author decision: collapse the text roles into one `text` kind carrying a `subtype`, leaving `memory_entry`, `code`, and the new `composite` as the other kinds.

**Why kind and layer became two different properties.** Folding the text roles into one kind raised a layer question, because the online-safety rule reads layer and an earlier draft put planner and monitor at L3. The author decision went further than "derive layer from subtype": planner and monitor are demoted to L1 text, and the L3 risk moves onto a `metacognition` composite (§18.2.1) that binds a planner, a monitor, and the memory/state the agent reasons over. The reasoning is that a standalone planner prompt is a low-risk wording change, while the metacognitive scaffold is dangerous precisely because the parts act together to modify the agent's own reasoning, so the risk is emergent from the composition rather than present in any constituent. Layer is therefore `max(subtype floor, max over constituents)`: text is L1, memory L2, code L4, and a composite can declare a floor (metacognition floors at L3) that exceeds its constituents. This is the cleaner answer to "where does metacognition live," and it makes Chapter 6 the chapter that introduces both metacognition and the composition machinery.

**Why a clean break rather than dual kinds.** The same reasoning as the Improver rename (§14): the codebase is pre-release with no production users, so carrying both the v0.1 per-role kinds and the v0.2 `text` subtypes would mean dual paths in every chapter and test for no benefit. Archives written under v0.1 are migrated on read by mapping each retired kind to `(text, subtype)`. The spec version bumps to 0.2 because the kind enum changed shape.

---

## 16. Artifact composability: factoring the joint search by a coupling graph

SPEC §16.1 settled "one improver per artifact," and §16.1.4 left a door open for a `MultiArtifactImprover` when artifacts are strongly coupled. The open question that door never answered: when an agent improves several artifacts, nothing accounts for which artifacts work better together, and the per-artifact improvers can each climb a local gradient into a joint saddle (§16.1.5). Composability is the answer, and it reframes the future extension: the coupling is declared in the data (a composite's constituents) rather than encoded in a special improver class.

**Why a composite is a real artifact, not just an improvement group.** The lighter option was an "improvement group" that lives only in a search config and is never stored. Making the composite a first-class Artifact (kind `composite`, content = constituent refs + binding) buys lineage, archive storage, replay, and a live champion for free from machinery that already exists. The decisive reason is deployment: a composite champion is the unit the agent deploys, which closes the gap §16.1.3 openly admits, that the deployed product of three separately-promoted artifacts was never evaluated as a unit. With a composite champion, the deployed combination is by construction a combination that was measured together.

**Why the graph factors the search.** The two naive options are "all artifacts independent" (today's default, no coordination) and "one global joint search over everything" (combinatorial, does not scale). The composition graph is the middle path: coupled artifacts form a cluster searched jointly under one composite signal, and everything uncoupled stays on cheap independent improvers. Across clusters the graph factors the problem; within a cluster, coordinate descent factors it again (improve one constituent at a time against the whole-bundle signal, holding the rest at champion), so the generic `ComposedSearch` is barely more expensive than a solo improver. Joint sampling (GRPO-style group selection over bundles) is the expensive opt-in reserved for strongly-coupled clusters.

**Why one whole-agent signal, optionally augmented.** A composite could be scored by combining per-constituent signals, but that quietly reintroduces the independence composition exists to remove. The constituents are coupled precisely so their joint effect on the end task is the thing that matters, so the default is a single whole-agent objective. A `CompositeSignal` (§3.5) may wrap it to fold in per-constituent metrics (skill token cost, memory latency) for multi-objective awareness, but the whole-agent score stays the arbiter.

**Why replace is enforced at construction.** Declaring a constituent inside a composite opts it out of solo improvement, and the framework refuses to construct a solo improver targeting a composite constituent. Documentation alone would let a solo improver and a composed search both mutate one artifact, which is the exact combination-drift fight composition is meant to end. Enforcing it at construction is the same earliest-possible-failure discipline as the online layer-safety check (§17.3), and it is why fix #2 (the layer-safety raise) should be designed to understand composite layers rather than patched in isolation.

---

## 17. Chapters 5 to 8: decisions from the planning interview (2026-08-23)

This section records why the decisions for these chapters came out the way they did, so a later session does not reopen them by accident.

**Why memory entries become archive artifacts, with the tier row keeping an `artifact_ref`.** Nothing in the framework can search a thing that is not an artifact, and Chapter 5 exists to search memory. Rewriting the tiers to store artifacts directly would throw away working SQLite code and the vector-store adapter boundary for no gain. So the tier row stays the retrieval index and points at the archive row, the same shape the procedural tier already uses, and UPSERTs become new versions with a parent.

**Why memory injection leaves `Agent.run` for constructor-declared hook wrappers.** The spec has said PRE_MODEL read and SESSION_END write since v0.1, and the inline code was a shortcut that two chapters now pay for. The first cost is cloning: an improver's `with_artifacts` clone must keep its memory behavior, and only wrappers that re-register their own hooks (the `Guardrail` pattern) survive a clone. The second cost is attribution: a utility signal cannot credit an entry the trajectory never recorded, so the read hook writes recalled-entry refs into `artifacts_used` as a side effect of being a hook.

**Why `UTILITY` is its own signal family rather than a reuse of `ENVIRONMENT_REWARD`.** An environment reward scores an answer: was the trajectory's outcome correct or approved. A utility measurement scores a memory entry's share of that outcome across the trajectories that recalled it, which is credit assignment, not grading. Overloading one kind for both would make `measurements_for_signal()` lie about what was measured; the union shape of `GapMeasurement` is untouched, so non-negotiable 3 holds.

**Why the group round stays in Chapter 5.** The framework has a pairwise round and an archive-sampling round hidden inside DGM, and GRPO-style selection is the obvious third shape: N candidates, one absolute signal, advantage over the group mean. Its natural target is insight extraction, where one trajectory yields several candidate lessons and only the above-average ones deserve to persist. Teaching it against memory entries keeps the RL framing of the chapter honest, and it is aimed at memory insights, never at tool-calling prompts.

**Why the reflector is a Signal, and why memory is its only route to behavior.** Reflection that edits the next run directly would need mutation rights at POST_MODEL or SESSION_END, and those rights are `None` in the hook table on purpose. A reflector that is a Signal writes its feedback to an L2 memory entry, and the only way that lesson changes a later run is the PRE_MODEL memory read. That makes the Huang et al. failure mode diagnosable: if the next trajectory's behavior diff is flat, the lesson was theater, and the chapter can say so with a number. It is also why Chapter 5 must precede Chapter 6.

**Why one coding agent serves Chapters 6 and 7.** Building a separate metacognition demo agent would add a second codebase the reader must hold while learning four new hooks. A coding agent built in v0 form in Chapter 6 gives planning and monitoring a place to visibly matter (which files to touch, a failing test caught before the answer ships), and the hidden-test score makes the behavior-change test exact rather than judged. Chapter 7 then grows the same agent, so familiarity compounds instead of resetting.

**Why the HITL gate is a bus subscriber, not a new hook point.** Non-negotiable 6 fixes the hook points to SPEC §6.2, and an approval checkpoint at archive commit is not one of them. The promotion design in §10 already emits `candidate_wins` after the archive records the candidate, and `auto_promote=False` already leaves the live champion untouched. A gate that subscribes to that event, writes a Proposal, and promotes on Decision needs no new seam, and it keeps the gate a review checkpoint rather than a suppression checkpoint.

**Why Chapter 7 ships a minimal gate and defers the ladder.** Chapter 7's job is to show code under search reaching a human safely: Proposal, Decision, a flat ApprovalPolicy, behavior diffs from replayed trajectories, and rollback as promoting the parent. The four-tier ladder, confidence routing, inoculation review, and the drift queue are about how humans spend attention across artifact kinds, which is Chapter 10's question. Splitting them keeps Chapter 7 about the harness and gives Chapter 10 something real to add.

**Why behavior diffs are a signal shared across chapters.** Chapter 6 needs a trajectory-versus-trajectory divergence measure to separate reflection theater from real change, and Chapter 7 needs the same measure on a proposal card. One `behavior_diff` signal returning `GapMeasurement` serves both, and it reuses the archive's trajectory table as its replay corpus instead of inventing a second store.

**Why planner code arrives in Chapter 7, not Chapter 6.** The planner is a text artifact at L1 in Chapter 6 because the chapter's subject is the scaffold, and the layer rule keeps the composite offline either way. Chapter 7 replaces the planner prompt with a planner function held as a CODE artifact, which is the first time a reasoning component is code. Putting that switch next to the sandbox and the test gate means the reader meets code-under-search once, with all its guards in view.

**Why HyperAgents gets its own Chapter 8.** Everything in Chapter 7 can run today: `TextDescriptionTool`, `ToolFromArtifact`, the composite, and the promotion path all exist, and the chapter's claims are checkable against the repo. Making the search method itself an artifact is a different promise, partly runnable (DGM's mutator prompt is plain text) and partly a thought experiment. Keeping it out of Chapter 7 lets that chapter stay shippable-today and lets Chapter 8 be honest about where the frontier stops.

**Why the meta level is read-only.** The layer rule says L1 and L2 may auto-promote online, L3 and L4 wait for a human, and a search method that edits its own mutator sits above all four. The outer loop's signal is the inner loop's gain, which is exactly the quantity a reward-hacking mutator would learn to inflate. Chapter 8 therefore runs its one meta hill-climb under a read-only policy with bounded mutation, and hands the reward-hacking runbook forward to Chapter 12.

## 18. Chapter 5 memory belongs to the harness

The harness installs memory tiers and connects them to the running agent through `MemoryHooks`. The learner targets that exact live tier and shares its archive, so an update is available to the next scoped recall. Persistence carries retained learning across sessions, while another agent must explicitly connect to the store and supply an authorized identity before it can use that learning.

Chapter 5 implements value-based reinforcement learning with a defined one-step update, `q_new = q + alpha * (credit - q)`. Learning changes usefulness estimates, while the update algorithm, credit rule, learning rate, retrieval weight, injection format, and hook timing remain configured. This distinction keeps improvement to memory separate from later work that searches or modifies the harness or the improvement procedure itself.

`MemoryLearner` makes that operation explicit without forcing the explanatory prose through the Search protocol. `MemRLSearch` inherits the learner and retains its existing adapter interface and archive labels for compatibility. The chapter examples and manuscript use the direct learner, and deterministic integration tests verify that changed values reach later requests through the existing hooks.

## 19. A studio support case makes episodic memory concrete

Second Breakfast Games is the approved Chapter 5 teaching scenario: recurring IT incidents around the fictional Moonbase Mayhem playtest. The corpus contains approved runbooks, while separate episodic entries retain attempts and observed outcomes. Maya's failed VPN reconnect is useful later context without becoming a successful fix merely because its utility rises.

The author's subsequent correction confines this scenario to practical demonstrations and exercises. Following Chapters 3 and 4, the chapter opens on memory and learning, then introduces the task where inputs, executable code, observed results, and memory records can be inspected. The studio and its characters are not an introductory story or a recurring narrative frame.

The walkthrough uses the existing Agent, MemoryHooks, MemoryLearner, and OnlineMemoryImprover with one recalled entry and a fixed lexical encoder. The optional encoder argument keeps the demonstration local without changing default memory behavior or adding a new framework primitive. The studio tools and simulator remain chapter application code, outside the core agent loop.

Scripted outcomes explain the mechanism and are labeled separately from measured model performance. The earlier research-helpdesk scripts and historical run evidence remain available, while the studio writes each execution into a new directory. Three additional draw.io figures connect the incident, memory versions, and feedback to the executable code.
