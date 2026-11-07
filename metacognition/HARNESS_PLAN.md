# Metacognitive Agent Harness: Build Plan

*Draft of 2026-11-03. Grounded in the chapter 6 study of* Self-Improving Agents *(`chapters/ch06`), the helix framework in `cxbxmxcx/self-improving-agents` at commit `fa7ec23`, and the OpenAI Agents SDK as documented for `openai-agents` 0.24.0. The package name `metacog` is a working name (see Decisions).*

## 1. Goal

Build a Python agent SDK that a developer who knows the OpenAI Agents SDK can pick up in minutes. Its agents should learn to think less. They turn repeated work into callable procedures and run familiar requests at the lowest reasoning setting measured to work. They also notice when a procedure goes stale and relearn it. None of this changes model weights.

The SDK makes two promises:

1. **Without metacognition it is an ordinary agent SDK.** It provides `Agent`, `Runner.run`, `@function_tool`, `ModelSettings`, sessions, guardrails, lifecycle hooks and tracing, in the shapes the Agents SDK uses.
2. **Metacognition is one argument.** The developer declares which procedures are worth learning and how a run is judged. The harness records runs, recalls procedures, chooses the reasoning setting, quarantines stale memory, relearns, and gates replacements.

## 2. Principles

These come from the study and from helix's SPEC. Each one rules out a design shortcut.

- **The loop never changes.** Every capability is a helix hook, memory tier, signal or search. The SDK compiles each run to helix's fixed loop (SPEC §6) and does not fork it.
- **The agent sees only two effects of metacognition:** a reasoning setting, and a set of recalled tools with one line of context each (study §3.7).
- **Procedures compute; the agent acts.** Learned code reaches read-only capabilities only. Side effects stay with the model's tool calls.
- **No verdicts, no learning.** Installing learned code requires an outcome signal the developer trusts. An agent without one records runs but never installs a procedure.
- **The measure must match deployment.** Gates and searches call a procedure the way the agent calls it, or with deliberately varied inputs (study §5.3, attempt B).
- **Crashes are crashes.** A procedure that errors is reported as an error, never as a rejection (study §5.4, first search).
- **Everything replays.** Every model call passes through a recording layer, and searches take explicit RNGs (study §7).
- **Trust is conditional.** A procedure stays trusted only while it keeps being accepted (study §5.6).

## 3. What helix provides, read as an Agents SDK

| Capability | helix today | Plan |
|---|---|---|
| Agent loop | `helix.agent.Agent.run(task: str, context) -> (output, Trajectory)`. LiteLLM Chat Completions, tools run one at a time, `max_iterations` | Use as is |
| Per-call settings | PRE_MODEL `RequestPatch(model, params)`. Later patches override earlier ones | ModelSettings, the controller's effort and `response_format` all travel on it |
| Tools | `helix.tools.tool`: a pydantic args model built from type hints | `@function_tool` wraps it, adding context injection, failure handling and a `read_only` flag |
| Guardrails | `Refusal` at PRE_MODEL or PRE_OUTPUT ends the run | `@input_guardrail` and `@output_guardrail` map to it and raise tripwire exceptions |
| Conversation input | One `task` string per run | Sessions inject history in a PRE_MODEL hook. `WorkingMemory.messages()` returns the live list, so no loop change is needed |
| Structured output | None | `response_format` through `RequestPatch`, parsed at PRE_OUTPUT. helix never loops back, so a bad parse raises an error |
| Streaming, handoffs, pause and resume | None. Sub-agents are tools (SPEC §11.9). PRE_TOOL can cancel a call | Step-level streaming from hooks and `Agent.as_tool()`. Handoffs, token streaming and pause/resume are out of scope for v1 |
| Tracing | `EventBus` events and OpenTelemetry spans. The Observatory reads SQLite archives | `trace()`, exporters and a metacognition panel |
| Memory | Episodic and procedural SQLite tiers with scopes. Code skills run under an AST guard | The metacognition store |
| Learning | `Contract`, `Episode`, `Writer`, five mechanisms, and chapter 7's `learn(lab)` runtime | Reused. Prompts are generalized |
| Measurement and search | `EffortSweep`, `ReasoningFloorSignal`, `BehaviorMatchSignal`, GEPA (evaluator, validate, rng), `CandidateSearch`, SQLite archive | The offline toolkit |
| Packaging | "Not a package": no installable pyproject, and heavy dependencies in `requirements.txt` | Decision 1 |

The chapter 6 online loop is not in helix. The recorder, controller, detector, gate, `act` and model-call recording live in `chapters/ch06/assistant/` and are tied to the calendar simulator. Generalizing those parts is the core of this work.

## 4. Developer experience

### 4.1 A plain agent (Agents SDK shape)

```python
from metacog import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

agent = Agent(name="Assistant", instructions="You are helpful.",
              tools=[get_weather], model="gpt-5.4-mini")
result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
```

Reasoning effort works the way it does in the Agents SDK: `ModelSettings(reasoning=Reasoning(effort="high"))` on the agent, with a run-level override through `RunConfig(model_settings=...)`.

### 4.2 Adding metacognition (the study's calendar assistant)

```python
from metacog import Agent, Runner, RunContextWrapper, function_tool
from metacog.meta import Metacognition, Verdict, procedure

@function_tool(read_only=True)        # learned procedures may call it through `api`
def read_calendar(ctx: RunContextWrapper[Calendar], person: str = "", day: str = "") -> dict:
    """Read one person's calendar for one day (mon to fri) ..."""
    return ctx.context.read_calendar(person, day)

@function_tool                        # side effects stay with the agent
def book_time(ctx: RunContextWrapper[Calendar], title: str, day: str, start: str,
              minutes: int, attendees: list[str] | None = None) -> dict:
    """Create an event on the user's calendar and invite attendees by full name ..."""
    return ctx.context.book_time(title, day, start, minutes, attendees)

@procedure(api=[read_calendar])
def find_shared_starts(attendees: list[str], days: list[str], minutes: int,
                       earliest: str = "", latest_end: str = "") -> dict:
    """Find every start time, by day, when the user and all the listed attendees
    are free for a meeting of the given length. ...

    Returns: {"starts": {day: ["HH:MM", ...]}}, earliest first.
    """
    ...   # no body: learn it. A real body makes it a hand-written, trusted memory.

async def checker(ctx: RunContextWrapper[Calendar], result) -> Verdict:
    verdict = ctx.context.check()
    return Verdict(accepted=verdict["success"], reason=verdict["reason"])

assistant = Agent(
    name="Maya's calendar assistant",
    instructions=PROMPT,
    tools=[read_calendar, book_time, cancel_event],
    model="gpt-5.4-mini",
    metacognition=Metacognition(
        store="runs/maya.meta",          # episodes, procedures, floors, lineage
        procedures=[find_shared_starts],
        verifier=checker,
        learner="value_learning",        # the mechanism the chapter 6 search kept
    ),
)

result = await Runner.run(assistant, "Book a 45-minute hiring debrief ...", context=calendar)
print(result.metacognition)   # effort='none', recalled=['find_shared_starts@4 trusted'], verdict='accepted'
```

What the harness does on each run:

1. **Recall.** Retrieve trusted procedures for the request (lexical encoder, `k=2`, relevance threshold). Bind each one as a tool whose `api` holds read-only proxies for this run's context, and add one line of context per procedure.
2. **Control.** Choose the reasoning setting by lookup. A recalled trusted procedure gets its measured floor; otherwise the run gets the no-memory floor or the agent's configured effort. The setting is applied as a `RequestPatch` on every model call.
3. **Run** helix's fixed loop.
4. **Record** the episode: request, tool steps, output, effort, procedures with their versions, usage, dollars and seconds.
5. **Judge.** Take the verifier's verdict now, or human feedback later through `await meta.feedback(run_id, accepted=False, reason=...)`.
6. **Detect.** For each procedure version, quarantine it when `limit` of its last `window` judged uses were rejected. Crashes are counted separately.
7. **Relearn** in the background. Every `N` runs after a quarantine (or after `M` accepted runs on a cold start), the learner writes a candidate from the episodes since quarantine.
8. **Gate.** Try the candidate on held-back cases at the adequate bar. On a pass, install it (or propose it for approval). On a failure, try again later.

### 4.3 Offline toolkit and CLI

```python
from metacog.offline import calibrate, search_learners, search_procedure

floors = await calibrate(assistant, cases=dev_cases, world=Calendar.for_case,
                         efforts=("none", "low", "medium", "high"), repeats=1)
floors.save("runs/maya.meta")                      # the controller's lookup table

memory = await search_procedure(find_shared_starts, cases=dev_cases, world=Calendar.for_case,
                                vary={"days": day_name_variants}, generations=4,
                                rng=random.Random(1165002))
kept = await search_learners(find_shared_starts, episodes=experience, reference=memory)
```

```text
metacog status runs/maya.meta        procedures, trust states, floors, savings
metacog show find_shared_starts -v 4 source, lineage, gate result, provenance
metacog approve <proposal>           when installs need a person
metacog replay <recording.jsonl>     re-run a recorded session without an API key
```

## 5. Architecture

### 5.1 Layers

```text
metacog (new package)
├─ SDK surface ...... Agent, Runner, function_tool, ModelSettings, RunConfig,
│                     RunResult, sessions, guardrails, lifecycle hooks, trace()
├─ metacognition .... procedures, recall, controller, recorder, verdicts,
│                     detector, learner, gate, practice, store, report
├─ offline toolkit .. calibrate, search_procedure, search_learners, CLI
└─ infrastructure ... recording/replay, effort ladders, prices, lexical encoder
        │  each run compiles to
        ▼
helix (book repo; its loop is not modified)
   Agent loop, HookRegistry, RequestPatch, Refusal, Tool, Trajectory,
   episodic and procedural tiers, code-skill guard, learning mechanisms,
   mechanism runtime, EffortSweep, signals, GEPA, CandidateSearch,
   SQLite archive, event bus
        │  LiteLLM
        ▼
model providers
```

### 5.2 How a run compiles to helix

| SDK concept | helix mechanism |
|---|---|
| `Agent.instructions` (string or callable) | System-prompt artifact, rendered per run |
| `@function_tool` | `helix.tools.Tool`. `RunContextWrapper` is injected and kept out of the schema |
| `ModelSettings`, `RunConfig.model_settings` | PRE_MODEL `RequestPatch` (temperature, reasoning effort, `response_format`, tool choice) |
| Controller effort | A later PRE_MODEL `RequestPatch`, so it overrides `ModelSettings` |
| Session history | PRE_MODEL hook that inserts history after the system message on the first call |
| Input and output guardrails | `Refusal` at PRE_MODEL or PRE_OUTPUT, raised as `InputGuardrailTripwireTriggered` or `OutputGuardrailTripwireTriggered` |
| `RunHooks`, `AgentHooks` | Adapters on SESSION_START, PRE_MODEL, POST_MODEL, PRE_TOOL, POST_TOOL and SESSION_END |
| `max_turns` | `max_iterations`. A run that ends as TIMED_OUT raises `MaxTurnsExceeded` |
| `RunResult` | Built from the trajectory: `final_output`, `new_items`, `raw_responses`, usage, cost, `to_input_list()` |
| Recalled procedures | Tools built before construction (`recalled_skill_tools`), plus the one-liner `MemoryHooks` writes |
| Tracing | Bus events and OpenTelemetry spans. `trace()` groups runs |

### 5.3 From the study to the SDK

| Study part | Where it lives today | SDK home | Change needed |
|---|---|---|---|
| Effort hook | `helix/eval/effort.py` (`EffortSetting`) | `metacog.meta.controller` | An effort ladder per model. Floors keyed by model and procedure |
| Callable procedural memory | `helix/memory/code_skills.py`, `procedural.py`, `memory/hooks.py` | `metacog.meta.procedures` | Trust states, versions, and an `api` bridge to async tools |
| Contract | `helix/memory/learning_mechanisms.Contract` | `@procedure` | Built from a typed stub |
| Recorder | `chapters/ch06/assistant/online.py` | `metacog.meta.recorder` | Deferred verdicts, cost, scopes |
| Controller | Same file | `metacog.meta.controller` | Relevance threshold and per-procedure floors |
| Detector | Same file | `metacog.meta.detector` | Persisted state and a separate crash channel |
| Learner | `learning_mechanisms.py`, `mechanism_runtime.py`, `chapters/ch07/mechanisms/` | `metacog.meta.learning` | Prompts that don't assume a domain. The calendar prompts ship as a pack |
| Gate and `act` | `online.py`, `chapters/ch06/assistant/learning.py` | `metacog.meta.gate`, `metacog.meta.practice` | No ground-truth arguments exist in production, so practice runs through the agent by default |
| Model-call recording | `chapters/ch06/assistant/recording.py` (`mock.patch`) | `metacog.recording` | A model-layer provider instead of patching |
| Floors | `EffortSweep`, `ReasoningFloorSignal` | `metacog.offline.calibrate` | Repeats per cell and a retrieval threshold |
| Context search | `helix/search/gepa.py`, `chapters/ch06/assistant/context_search.py` | `metacog.offline.search_procedure` | Input variation built in |
| Mechanism search | `helix/search/candidates.py`, `signals/behavior_match.py` | `metacog.offline.search_learners` | None |
| Lexical encoder | `chapters/ch05/personal/encoder.py` | `metacog.retrieval` | The default. Embeddings become an optional extra |

### 5.4 The metacognition store

One directory per agent deployment holds:

- the episodic tier (episodes with verdicts, effort, procedures used and cost);
- the procedural tier (procedure versions with trust state: candidate, trusted, quarantined or retired);
- a helix SQLite archive (lineage, measurements, gate results, proposals);
- the floor table (JSON);
- detector state.

Because it uses helix's own stores, the Observatory can read it.

## 6. Design decisions

1. **Compile to helix per run.** The SDK `Agent` is a configuration dataclass, as in the Agents SDK. `Runner` builds a helix `Agent` for each run because recalled procedures bind to that run's context; the study's `build_agent` does the same. helix itself changes only additively, and only when a hook cannot do the job.
2. **Procedures are typed stubs.** `@procedure` reads the name, signature, docstring (description and `Returns:`) and `api=[...]` to build helix's `Contract`, and prepends the bound `api` parameter. A `...` body means "learn it"; a real body is a hand-written trusted memory, like the book's `find_shared_starts.py`. Learned and searched versions get parent pointers in the archive.
3. **Read-only capabilities for learned code.** `api` exposes only tools marked `read_only=True`. They appear as synchronous proxies that run the async tool on the serving event loop with this run's context. Learned code runs in a worker thread, as `code_skills.skill_tool` already does.
4. **Practice environments.** Learners and gates must try a candidate on a past request. The study's `act` calls the function with the simulator's ground-truth arguments, which production does not have. There are two modes:
   - **Agent-level (default).** Run the agent at the memory setting with the candidate installed, in a world rebuilt for the case, and ask the verifier. This measures the procedure as it is used: attempt B's lesson. In the study's terms it costs about $0.002 per case at `none`.
   - **Function-level.** Call the function directly when cases carry contract arguments (simulators, fixtures). It is cheaper, and the calendar parity example uses it.

   Worlds come from a developer factory (`world=lambda case: ...`) in M2. M4 adds worlds rebuilt from recorded read-only results, with side effects dry-run.
5. **Controller.** A lookup table maps (model, procedure) to an effort, built by calibration. Without calibration, a trusted procedure gets the lowest rung and no procedure gets the agent's configured effort. Effort ladders per model family map rungs to provider parameters: `reasoning_effort` for GPT-5.x through LiteLLM, and other families only with tests. An optional mode, off by default and marked experimental, escalates to the no-memory setting for the rest of a run when a procedure call errors.
6. **Verdict sources.** The sources are a programmatic verifier, deferred human feedback, or a judge. Each verdict records its source. The detector and gate act only on sources configured as gating (by default the verifier and human feedback). Judges can inform the detector but cannot install code.
7. **Detector.** The study's hand-set defaults apply per procedure version: `window=4`, `limit=2` over judged uses. State persists across restarts. Crashes and timeouts get their own counter, which fixes E3's misreported crashes.
8. **Learner.** Learners use chapter 7's interface, `async def learn(lab)`, with a metered lab (writer calls, practice calls, seconds). Built-ins:
   - value learning (the default);
   - reflection;
   - success distillation;
   - transition distillation;
   - skill synthesis;
   - the no-experience control.

   Prompts become templates filled from the agent and the contract. The book's calendar wording ships as a prompt pack so its replays stay byte-identical.
9. **Gate.** Held-back cases come from a developer gate set or from recent accepted episodes the learner never saw. The bar is adequate (7 in 8) with at least 6 cases. A pass installs the procedure, or creates a proposal when `approval="human"`, which is the default when the verifier is not programmatic. A newly installed procedure starts with a fresh detector window.
10. **Execution.** M2 runs learned code under helix's in-process guard (AST check, builtin allow-list, `range` cap, deadline), which is labeled as not a sandbox. M4 adds a subprocess executor with resource limits that proxies `api` calls over a pipe, and makes it the default for model-written code.
11. **Recording and replay.** A model-layer `Recording` (record, replay or live) is keyed the way the book keys calls: model, messages, tools, `reasoning_effort` and `temperature`. It is set with `RunConfig(recording=...)` or globally, and writers and searches use it too. Searches take an explicit RNG.
12. **Learning stays off the serving path.** Each procedure gets one background learning task at a time, with an inline mode for tests and replays. Practice and gate runs are tagged and never become experience.
13. **Retrieval.** The default is the lexical encoder, which avoids helix's 440 MB embedding download. `k=2` is the default because in E4, `k=1` ranked the wrong function first for 4 of 16 requests. Calibration sets the relevance threshold.
14. **A narrow harness port.** The metacognition core talks to the loop through four calls: bind tools, add context, patch effort, and receive the run record. helix implements the port now. A later adapter could implement it for OpenAI Agents SDK agents through `RunHooks`, `ModelSettings` and function tools.

## 7. Milestones

Every milestone ends with tests that run without an API key.

**M0 Foundations**
- Confirm that the book's chapter 6 listings replay E1, E3 and E5 in a clean Python 3.12 venv with no API key.
- Settle Decision 1 and create the repository skeleton: src layout, pyproject, pytest, ruff, `examples/`, `docs/`.
- Port the recording provider, lexical encoder and price table into `metacog`.
- Copy the calendar world, generators, memories and recordings (3.4 MB, Apache-2.0) into `examples/calendar/`.
- *Accept:* E1's sweep, run through helix and `metacog.recording` in replay, reproduces 2, 4, 12 and 13 of 16 without memory and 16 of 16 with the searched memory.

**M1 SDK core (no metacognition)**
- `Agent`, `Runner.run` and `run_sync`, `RunConfig`, and `RunResult` (`final_output`, `new_items`, `raw_responses`, usage, cost, trajectory, `to_input_list`).
- `@function_tool` with sync and async functions, context injection, name and description overrides, `failure_error_function` and `read_only`.
- `ModelSettings` and `Reasoning`, `output_type`, input and output guardrails, `RunHooks` and `AgentHooks`, `SQLiteSession`, `trace()` and `Agent.as_tool()`.
- A quickstart and an API reference.
- *Accept:* the Agents SDK quickstart runs with only the import changed. The calendar agent built on the SDK surface replays E1 with identical request hashes.

**M2 Metacognition runtime**
- `@procedure` and the procedure store (trust states, lineage).
- Recall and the `api` bridge.
- The controller (floor table and effort ladders), the recorder, verdict sources and `feedback()`.
- The detector, the learner (lab, built-ins, prompt packs), the gate (holdout and approval), and practice (both levels, factory worlds).
- `RunResult.metacognition`, bus events, and `meta.report()` with the study's economics: cost per phase, learning spend, break-even.
- *Accept:* the calendar example replays E5. The memory is quarantined after run 8, then relearned at run 12 from 4 runs with 3 writer calls. The gate passes 6 of 6, and all 8 later requests are accepted at `none`. Relearning costs $0.18, breaking even after about 7.6 requests. Each part also has unit tests that need no replay.

**M3 Offline toolkit and CLI**
- `calibrate`: sweeps, solid and adequate floors, repeats, and the retrieval threshold, written to the floor table.
- `search_procedure`: GEPA over code, with input variation and execution traces in its feedback.
- `search_learners`: `CandidateSearch` with `BehaviorMatchSignal`.
- The `metacog` CLI.
- *Accept:* replays reproduce E2 attempt D (v15 kept, 16/16 validation and test), the E3 second-search table, and E4 (`k=2` serves both task classes).

**M4 Production and platform**
- A subprocess executor, the default for model-written code.
- Worlds rebuilt from recordings, with side effects dry-run.
- End-to-end human approval and per-user scopes.
- Concurrency safety: one writer, and one learning task per procedure.
- Step-level `run_streamed` and an OpenTelemetry exporter.
- A metacognition panel in the Observatory (`helix_web`): floors, trust timeline, lineage, savings.
- *Accept:* learned code that crashes, runs slowly or behaves hostilely is contained. A restart in the middle of a quarantine resumes correctly. The Observatory shows the whole E5 timeline.

**M5 New evidence (live, paid)**
- A second domain whose conventions can't be inferred, run end to end on the SDK.
- Repeated trials (at least 3 per configuration) to address the study's one-sample limitation.
- A comparison of agent-level and function-level gates.
- *Budget:* similar to the chapter 6 study, whose recordings cost about $14.50 in total.

## 8. Risks and open questions

1. **Replay parity is brittle.** Any change to prompt text, tool schemas or message order changes request hashes. Parity examples must use the book's exact strings, through prompt packs and helix's own schema generation. Re-recording costs money.
2. **Verdicts are scarce.** Most real deployments have no checker, so the loop will only record. Implicit signals and judges help with detection but must not install code.
3. **Wrong recall at low effort.** A request that only resembles a procedure's class runs at `none`. The mitigations are the calibrated threshold, `k=2` and the optional escalation.
4. **Learned-code safety.** helix's own docstring says the guard is not a sandbox. The M4 subprocess executor must ship before any untrusted use.
5. **Limits on matching the Agents SDK.** helix uses Chat Completions through LiteLLM, runs tools one at a time, and neither streams tokens nor hands off. The SDK will match the Agents SDK's shapes but not its wire format. Input is chat messages, not Responses items. Function-tool schemas follow helix (non-strict by default, `strict_mode=True` opt-in). Responses-only features such as hosted tools and reasoning items across turns are out of scope.
6. **Learned functions overfit.** They encode the wording they saw: `"half an hour"` became 30, while "45 minutes" would get no buffer. The detector catches this after failures; input variation in gates and searches makes it less likely.
7. **Upstream drift.** helix changes with each chapter (additively, under SPEC §14). Pin a commit and run the parity suite on every upgrade. SPEC §11.6 still describes the earlier Router, Monitor and Reflector scaffold rather than the study's loop, so the SPEC needs an update for chapter 6.

## 9. Status (2026-11-03)

M0 to M5 are built in this repository, with 71 tests that need no API key.

- **Parity.** The book's E1, E2, E3, and E5 replay through metacog with the study's numbers:
  - E1 through `calibrate` and the SDK surface;
  - E2 through `search_procedure` (v15 kept);
  - E3 through `search_learners` (the full mechanism table);
  - E5 through `Metacognition` (quarantine at run 8, relearn at 12, gate 6/6, $0.1805, break-even 7.6).
- **Deviations from the plan.**
  - helix comes from a checkout linked by `scripts/link_helix.py`; the book repository is unchanged (Decision 1 is still open).
  - The Observatory panel became a standalone HTML report (`metacog report --html`) plus `MetacognitionEvent`s on helix's bus, so `helix_web` is unchanged.
  - Recording routes calls per context rather than patching module names; the book's keys are kept.
- **M5 so far.** The expense world, its agent, and `examples/expenses/study.py` are built. A small live check (4 development trips, 49 calls, $0.06):

  | Arm | `none` | `high` |
  |---|---|---|
  | No memory | 0/4 | 2/4 |
  | Hand-written procedure | 4/4 | 4/4 |

  This is the study's pattern on a second domain, but four trips is far too few to conclude anything.
- **Still to run.** The full M5 study (16 trips per split, learning with the GENERIC prompts, the online loop through the policy change, and repeated trials) is left for the testing phase. Its budget is about $15.

## 10. Decisions needed

1. **Where the SDK lives and how it gets helix.** *Settled on 2026-11-04: in the book repository, as the side project `metacognition/` (see ARCHITECTURE.md §8).* The original recommendation was a new repository in `C:\AI-workspace\metacognition` that depends on helix. To make that possible, add `[project]` packaging to the book repo's `pyproject.toml` with helix's core dependencies (litellm, pydantic, opentelemetry, httpx, python-dotenv), put the UI, retrieval and embedding dependencies in extras, and pin to a commit. That change contradicts the README's "This is not a Python package." There are two alternatives:
   - copy the needed helix modules into the SDK, leaving the book repo untouched but forking the code;
   - build the SDK as a new top-level package inside the book repo.
2. **Package name.** `metacog` is a placeholder; check PyPI availability before M0.
3. **Default install policy.** Recommended: install automatically after the gate only when the verifier is programmatic; otherwise require approval.

## 11. Phase 2

Phase 2 makes the harness fit for building and deploying real agents, then uses it on example domains. Its architecture, the review findings it fixes, its milestones (A1 to A4) and the domains (D1 to D4) are in [ARCHITECTURE.md](ARCHITECTURE.md).
