# metacog

An agent SDK shaped like the OpenAI Agents SDK whose agents learn to think less. Each run compiles to the
helix agent loop from *Self-Improving Agents* (Micheal Lanham, Manning). Metacognition (callable procedures,
reasoning floors, and the online recorder, controller, detector, learner, and gate from chapter 6) is added
as hooks around that loop, never as a change to it.

This folder is a side project of the book's companion repository: a self-contained, installable harness that
uses the repository's `helix/` as its agent loop and replays chapter 6's recordings as its regression suite.

- From design to drift, step by step, on the expense example: [GUIDE.md](GUIDE.md)
- Design and milestones: [HARNESS_PLAN.md](HARNESS_PLAN.md)
- Architecture and phase 2 (deployment, then example domains): [ARCHITECTURE.md](ARCHITECTURE.md)
- The research it builds on: chapter 6's study, [`chapters/ch06`](../chapters/ch06/README.md)

Status: phase 1 (milestones M0 to M5) is built and tested. The study's experiments E1, E2, E3, and E5 replay
through metacog from the book's recordings with the study's numbers. A second task class (expense claims) and
its study script are in `examples/expenses`; only a small live check of it has been run so far. Phase 2
readies the harness for real deployments, and its architecture work is done:
- A1: agents with several procedures, on several models;
- A2: deployment (a learning worker, a shared store, sandboxing by default, a budget, a judge);
- A3: case files and `metacog calibrate`, `metacog discover`, webhooks, and `metacog serve` with a live dashboard;
- A4: CI (`.github/workflows/metacognition.yml`) and the guide.

The example domains are under way ([DOMAINS.md](DOMAINS.md)):
- **D1, the full expense study,** has run live: once through, then its measurements three times over, for $2.00
  in all. It reproduced the study's results on a second domain, and the repeats show a procedure removes
  variance as well as cost. Its recording replays offline.
- **D2, a returns desk where verdicts are scarce, late and human,** has run live twice, plus follow-ups, for
  $3.40 in all.
  - **First pass:** without a checker, the procedure stayed stale longest and cost the most in wrong refunds.
    A judge on every run caught problems fast but raised false alarms.
  - **Second pass:** it applied the first pass's lessons. The procedure answered in its action tool's terms,
    the reasoning settings came from measured floors, and the gate checked 6 labeled cases. Wrong actions fell
    from 13 to 3 with a checker and from 11 to 4 with a judge.
  - **Follow-ups** fixed the two problems the second pass found. Telling the agent to act on a procedure's
    answer stopped it overruling a right one (76 of 76 returns at `none`, a third cheaper). Letting one
    rejection from a person quarantine (`Detector(limits={"human": 1})`) cut the scarce-review arm's wrong
    actions from 27 to 15.
- **D3, invoice intake with several procedures and several clients,** is built and tested offline, and has not
  run live yet. It found three things the harness can't yet set per procedure: its scope, its gate and its
  learner.

Learning about learning (A5, [DOMAINS.md](DOMAINS.md)) has run live for $2.10.
- **Tournaments:** learner tournaments on D1 and D2 found no learner best on both, so the loop picks one per
  procedure.
- **A designed learner:** a design search wrote a new learning mechanism (`examples/learning/designed/`) that
  learns both with the cheapest writer, which no existing mechanism did.
- **Transfer:** on D3's invoices, which it never saw, it learned 7 of 12 intakes and 12 of 16 payments, where
  every existing mechanism learned at most 2 and 0.
- **Repeats:** two repeated searches varied widely, and only one transferred.
- **The worker designs:** when no mechanism can learn a procedure, the worker now designs one for a person to
  approve.

The suite has 190 tests and needs no API key. Nothing here is stable yet.

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
world with side effects dry-run), `learners` (a learner per procedure; any learner can be a `SourceLearner`,
chapter 7's form of a mechanism held as source and run through helix's metered lab, such as
`SourceLearner.seed("skill_synthesis")`), `selection` (a `LearnerSelection`: when learning keeps failing for a
procedure, a tournament of mechanisms on its own runs picks its learner), `gate` (held-out case ids or the last accepted runs; with its own `practice`
when it checks labeled cases another way than the learner practices), `detector`, `controller`, `router`,
`attribution`, `approval`
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
  controller reads them on every run. `metacog calibrate` does it for you (see below).
- **Tell someone.** `metacog serve ... --webhook URL` and `metacog worker ... --webhook URL` post quarantines,
  proposals and installs to a URL as JSON, with a one-line `text` a chat channel can show.

## The service and the live dashboard

`metacog serve app:agent` hosts an agent's deployment over HTTP. At `/` it serves a dashboard that updates as
runs land:

- **Needs you:** versions the gate proposed, with their diff, gate result and cost, and an Approve button;
  runs awaiting a person's verdict, with Accept and Reject and a reason.
- **Runs:** the procedures each run was offered and called, its reasoning setting, verdict and cost. Click a
  run for its request, routing scores, tool calls with results, answer, and why it was judged as it was.
- **Procedures:** each one's trust state, its last four judged uses, its versions, and a Quarantine button.

Everything it shows is in the store and everything it does is an API call, so it adds no capability the CLI
lacks; it puts a person in the loop as the loop runs. To see it with no API key, replay chapter 6's drift
with a person approving installs:

```powershell
.venv\Scripts\python -m examples.calendar.drift_demo     # E5's first 12 runs, into examples/runs/drift.meta
.venv\Scripts\metacog serve examples.calendar.drift_demo:agent
# open http://127.0.0.1:8620: v1 quarantined at run 8, v2 relearned at run 12 and waiting for you
```

The API (`/api/docs` lists it) is `POST /api/runs` to run the agent, `POST /api/runs/{id}/feedback`,
`GET /api/queue`, `GET /api/procedures`, `POST /api/procedures/{name}/{version}/approve`,
`POST /api/procedures/{name}/quarantine`, and `GET /api/stream` for live updates. A run's context comes from
`--context module:attribute`, a factory that builds it from the request body. Every POST needs the header
`X-Metacog: 1`, which the dashboard sends and another site's page can't, and there is no other authentication:
keep the default localhost binding, or put the service behind your own. It needs the `serve` extra
(`pip install -e ".[serve]"`).

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
metacog runs runs/jordan.meta            recent runs: model, effort, procedures offered and called, verdict
metacog show runs/jordan.meta NAME       a version's source and provenance
metacog mechanisms runs/jordan.meta       designed mechanisms, and the learner a tournament chose per procedure
metacog approve-mechanism runs/jordan.meta NAME   let a designed mechanism compete in learner tournaments
metacog approve runs/jordan.meta NAME V  trust a version the gate proposed
metacog quarantine runs/jordan.meta NAME stop trusting a procedure
metacog report runs/jordan.meta --html report.html
metacog discover app:agent [--draft]     recurring, costly work without a procedure: candidates for one
metacog feedback app:agent 12 reject --reason "..."    a person's verdict on run 12
metacog calibrate app:agent cases.jsonl --world app:world_for    floors (and routing) into the store
metacog serve app:agent                  the API and the live dashboard
metacog worker app:agent [--once]        learn for an agent with learning="worker"
```

Every command that takes a store also takes a target such as `app:agent`. A case file has one JSON object a
line: `{"id": ..., "input": ..., "data": ..., "procedure": ...}`. `data` is what `--world` rebuilds the case's
context from, and `procedure` (a name, or `null` for a request in no procedure's class) labels the case for
routing calibration. `metacog calibrate --recording calls.jsonl` reuses recorded replies and records new ones.

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
recordings from `chapters/ch06`. The repository's own `pytest` (run from its root) doesn't collect them; the
`metacognition` GitHub workflow runs them whenever this folder, `helix/` or `chapters/ch06/` changes.
`constraints.txt` pins the versions the suite last passed with (`pip install -c constraints.txt -e ".[dev]"`);
refresh it with `uv pip freeze --exclude-editable` after an upgrade passes. For live calls, put provider keys
in a `.env` file in this folder (it is gitignored). Record new runs with `Recording(path, mode="record")` or
`RunConfig(recording=...)`.

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
                      routing, case files, searches), discover, service and dashboard.html, cli, testing
src/metacog/meta/     metacognition: procedure, runtime, parts (router, controller, detector, gate, budget),
                      mechanisms, prompts, store, api, sandbox, replay, scopes, events, judge, worker, report
examples/calendar/    chapter 6's assistant on metacog, and the drift demo for the dashboard
examples/expenses/    the second task class (D1): world, agent, app, study, recorded runs
examples/refunds/     D2's returns desk: world, agent, study with reviewers, a judge and replayed practice
examples/invoices/    D3's invoice intake: two procedures, three clients, routing, discover, and scopes
examples/learning/    A5 live: learning problems from the domains, learner tournaments, and a design search
tests/                unit tests, and parity tests that replay the study's experiments
```
