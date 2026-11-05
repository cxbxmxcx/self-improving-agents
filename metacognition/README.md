# metacog

An agent SDK shaped like the OpenAI Agents SDK whose agents learn to think less. Each run compiles to the
helix agent loop from *Self-Improving Agents* (Micheal Lanham, Manning). Metacognition (callable procedures,
reasoning floors, and the online recorder, controller, detector, learner, and gate from chapter 6) is added
as hooks around that loop, never as a change to it.

This folder is a side project of the book's companion repository: a self-contained, installable harness that
uses the repository's `helix/` as its agent loop and replays chapter 6's recordings as its regression suite.

- Design and milestones: [HARNESS_PLAN.md](HARNESS_PLAN.md)
- Architecture and phase 2 (deployment, then example domains): [ARCHITECTURE.md](ARCHITECTURE.md)
- The research it builds on: chapter 6's study, [`chapters/ch06`](../chapters/ch06/README.md)

Status: phase 1 (milestones M0 to M5) is built and tested. The study's experiments E1, E2, E3, and E5 replay
through metacog from the book's recordings with the study's numbers. A second task class (expense claims) and
its study script are in `examples/expenses`; only a small live check of it has been run so far. Phase 2, which
readies the harness for real deployments, is under way: A1 (agents with several procedures, on several models)
and A2 (deployment: a learning worker, a shared store, sandboxing by default, a budget, a judge) are built.
The suite has 100 tests and needs no API key. Nothing here is stable yet.

## A plain agent

```python
from metacog import Agent, Runner, function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"

agent = Agent(name="Assistant", instructions="You are helpful.", tools=[get_weather], model="gpt-5.4-mini")
result = Runner.run_sync(agent, "What's the weather in Tokyo?")
print(result.final_output)
```

`Agent` (`clone`, `as_tool`), `Runner.run` / `run_sync` / `run_streamed`, `RunConfig`, `RunResult`
(`final_output`, `new_items`, `raw_responses`, `usage`, `cost`, `trajectory`, `to_input_list`),
`@function_tool`, `ModelSettings` / `Reasoning`, `output_type`, input and output guardrails, `RunHooks` /
`AgentHooks`, `SQLiteSession`, and `trace()` work as they do in the Agents SDK, with the differences listed
below.

## An agent that learns to think less

Declare the procedure worth learning as a typed stub, say how a run is judged, and attach metacognition:

```python
from metacog import Agent, Runner, RunContextWrapper, function_tool
from metacog.meta import Metacognition, procedure

@function_tool(read_only=True)          # learned procedures may call read-only tools through `api`
def read_trip(ctx: RunContextWrapper[Expenses], trip_id: str) -> dict:
    """Read a trip's details and every receipt."""
    return ctx.context.read_trip(trip_id)

@function_tool                          # side effects stay with the agent
def file_claim(ctx: RunContextWrapper[Expenses], trip_id: str, amount_usd: float) -> dict:
    """File the expense claim for a trip, in US dollars."""
    return ctx.context.file_claim(trip_id, amount_usd)

@procedure(api=[read_trip, read_policy])
def reimbursable_total(trip_id: str) -> dict:
    """Work out how much of a trip's receipts the company reimburses, in US dollars.

    Returns: {"total_usd": float}, rounded to the cent.
    """
    ...                                 # no body: learn it (a real body is a hand-written, trusted memory)

meta = Metacognition([reimbursable_total],
                     verifier=lambda ctx, result: ctx.context.check(),   # or feedback later: meta.feedback(...)
                     practice=practice, store="runs/jordan.meta")
agent = Agent(name="Expense assistant", instructions=PROMPT, tools=[read_trip, read_policy, file_claim],
              model="gpt-5.4-mini", metacognition=meta)

result = await Runner.run(agent, "File my claim for trip T-104.", context=expenses)
print(result.metacognition)   # effort=none offered=['reimbursable_total@2'] verdict=accepted
```

Each run, metacognition recalls trusted procedures (each offered to the model as a tool with one line of
context), chooses the reasoning setting by lookup, records the run, and judges it. A detector quarantines a
procedure whose recent uses keep failing; the learner writes a replacement from the runs since, and the gate
installs it only when it clears the bar on cases it never learned from. The agent itself sees only two
effects: a reasoning setting and a set of tools.

Options: `learner` (value learning by default, or reflection, success or transition distillation, skill
synthesis), `prompts` (`GENERIC`, or the book's `CALENDAR` for replay), `practice` (a function, an
`AgentPractice` that runs the agent in a rebuilt world, or a `ReplayPractice` that replays an accepted run's
world with side effects dry-run), `gate`, `detector`, `controller`, `router`, `attribution`, `approval`
("auto" or "human"), `executor` ("guarded" in-process, or "subprocess" for model-written code), and `PerUser`
for one procedure set per person.

With more than one procedure, three things matter:

- **Routing.** Without thresholds every request matches the best procedures, which suits an agent with one
  kind of request. Label some requests, including ones that belong to no procedure, and calibrate:
  `router = calibrate_routing(procedures, labeled).router`. A request that matches nothing runs at the
  agent's own setting and teaches no procedure anything.
- **Blame.** A rejected run counts against the procedures it called (`attribution="called"`), not everything
  it was offered. `attribution="offered"` is the study's rule.
- **The reasoning setting.** When several procedures are offered, the run gets the most demanding of their
  floors.

Floors belong to a model. `calibrate` records the model it measured, and the controller uses those floors
only for that model; a model without floors runs at the agent's own setting until it is calibrated. Settings
reach each provider through its effort ladder (`metacog.ladders`). OpenAI and Anthropic models both take
LiteLLM's `reasoning_effort`, which LiteLLM turns into Claude's thinking settings.

## Deploying an agent

- **Learn in a worker.** With `learning="worker"`, serving processes only record, judge and quarantine.
  `metacog worker app:agent` learns for the agent that `app.agent` names. It holds a lease on the store, so
  only one worker learns for a store, and an attempt cut short by a crash is retried by the next worker.
  `"background"` (an asyncio task beside serving) and `"inline"` (for tests and replays) remain.
- **Share the store.** On disk the store uses SQLite's write-ahead log and waits for locks, so serving
  processes, the worker and the CLI on one host can use it at once. Stores from earlier versions migrate when
  opened.
- **Run model-written code in a child process.** `executor="auto"`, the default, runs procedures people wrote
  in-process under helix's guard and model-written ones in a fresh `python -I` child process.
- **Cap learning.** `budget=LearningBudget(attempts_per_day=3, dollars_per_day=1.0)` limits each procedure.
  A skipped attempt is logged with the reason.
- **Judge without a checker.** `verifier=Judge("The claim is the policy total, in US dollars.")` has a model
  read each run against a rubric. Its verdicts are recorded, but they reach the detector only if `"judge"` is
  in `gating_sources`, and code learned under a judge waits for a person's approval.
- **Keep floors in the store.** `floors.record(meta)` writes calibrated floors for their model, and the
  controller reads them on every run.

## Offline toolkit and CLI

```python
from metacog.offline import calibrate, calibrate_routing, search_learners, search_procedure

floors = await calibrate(agent, cases, verifier=check, world=Expenses)   # reasoning floors per arm, one model
floors.save("runs/jordan.meta")                                           # Controller.from_floors reads it
routing = calibrate_routing(procedures, labeled)                          # thresholds from labeled requests
found = await search_procedure(reimbursable_total, practice=practice, cases=ids, variants=[...], rng=rng)
kept = await search_learners(reimbursable_total, episodes, practice=practice, reference=src, run_case=run, cases=cases)
```

```text
metacog status runs/jordan.meta          procedures, trust states, and economics
metacog runs runs/jordan.meta            recent runs: effort, procedures offered, verdict, cost
metacog show runs/jordan.meta NAME       a version's source and provenance
metacog approve runs/jordan.meta NAME V  trust a version the gate proposed
metacog quarantine runs/jordan.meta NAME stop trusting a procedure
metacog report runs/jordan.meta --html report.html
metacog worker app:agent [--once]        learn for an agent with learning="worker"
```

Every command that takes a store also takes a target such as `app:agent`.

## Development setup

Unlike the rest of the repository, this project is pip-installable, in its own environment. helix is not, so
a small script links the repository root (this folder's parent) onto the environment's import path.

```powershell
cd metacognition
uv venv --python 3.12 .venv                    # or: python -m venv .venv
uv pip install --python .venv -e ".[dev]"      # or: .venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\python scripts\link_helix.py     # finds the repository root above this folder
.venv\Scripts\python -m pytest
```

Tests replay recorded model calls and need no API key; the parity tests replay the book's chapter 6
recordings from `chapters/ch06`. The repository's own `pytest` (run from its root) doesn't collect them. For
live calls, put provider keys in a `.env` file in this folder (it is gitignored). Record new runs with
`Recording(path, mode="record")` or `RunConfig(recording=...)`.

The expense study (`python -m examples.expenses.study --help`) replays its committed smoke recording by
default; `--mode record --budget N` makes paid calls only for replies the recording lacks.

## Differences from the OpenAI Agents SDK

By design (HARNESS_PLAN.md §8): calls go through LiteLLM chat completions, so messages are chat dicts rather
than Responses items; tools run one at a time; streaming is step-level (no token deltas); there are no
handoffs or pause-and-resume approvals; tool schemas follow helix (non-strict); a tool error reaches the
model as `{"error": ...}` unless the tool sets `failure_error_function`.

## Layout

```text
src/metacog/          the SDK: agent, run, tool, model_settings, ladders (effort per model family), guardrail,
                      lifecycle, session, stream, tracing, recording (record/replay), offline (calibrate,
                      routing, searches), cli, testing
src/metacog/meta/     metacognition: procedure, runtime, parts (router, controller, detector, gate, budget),
                      mechanisms, prompts, store, api, sandbox, replay, scopes, events, judge, worker, report
examples/expenses/    the second task class: world, agent, study, recorded smoke run
tests/                unit tests, and parity tests that replay the study's experiments
```
