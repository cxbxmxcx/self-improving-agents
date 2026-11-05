# metacog architecture

*2026-11-04. Phase 2 of [HARNESS_PLAN.md](HARNESS_PLAN.md). Phase 1 (milestones M0 to M5) built a harness that
replays the chapter 6 study. Phase 2 turns it into a harness for building, deploying and operating real agents,
then uses it on example domains to explore metacognition. Everything below was checked against the code at
commit 568050f of the standalone repository the harness grew in, before it moved here.*

## 1. What the harness is for

A developer builds an agent the way they would with the OpenAI Agents SDK. metacog adds one thing: the agent's
own performance becomes something the harness measures and manages. It learns which recurring work can be done
by code, how much reasoning each kind of request needs, and when what it learned has gone stale. It changes
neither the prompt nor the model.

| Piece | OpenAI platform | metacog | Today |
|---|---|---|---|
| Build | Agents SDK | `Agent`, `Runner`, `@function_tool`, guardrails, sessions, streaming | Built |
| Evaluate | Evals | Cases, verifiers, `calibrate` (reasoning floors) | Python only: no case files, judge or command |
| Observe | Traces dashboard | OpenTelemetry spans, bus events, the store, `metacog report` | A static HTML report |
| Deploy | ChatKit, Agent Builder | Nothing yet | No way to host an agent |
| Improve | Prompt optimizer, fine-tuning | Metacognition: procedures, controller, detector, learner, gate, approval | Works for one procedure on one model family |

## 2. Metacognition across an agent's life

Each stage of building and running an agent raises a question about the agent's own thinking. The harness
should answer each one with a tool.

| Stage | The question | Tool | Today |
|---|---|---|---|
| Design | Which recurring work deserves a procedure? | `@procedure` stubs; `metacog discover` finds repeated, costly work in recorded runs | Stubs built; `discover` is new |
| Evaluate | How much reasoning does each kind of request need on this model, with and without a procedure? | Cases and a verifier, then `calibrate`, which writes the floors | Built for one model family |
| Bootstrap | Where does the first procedure come from? | A hand-written body, `search_procedure`, or online learning from a cold start | Built |
| Release | Can this code be trusted? | The gate on held-back cases, approval, the subprocess executor | Built, but model-written code runs in-process by default |
| Operate | Is the agent still right, and what does it cost? | The recorder, verdicts (verifier, person, judge), the economics report | No judge and no live view |
| Adapt | Has the world changed? | Detector, quarantine, relearning, gate | Built for one procedure |
| Revisit | Has the model changed? | Recalibrate the floors for the new model | Floors aren't keyed by model |

## 3. Architecture

### 3.1 Layers

| Layer | Contents | Phase 2 change |
|---|---|---|
| Operations | The `metacog` CLI, a service (HTTP), a dashboard | The service and dashboard are new, as an optional extra |
| Lifecycle toolkit | Case sets, graders, `calibrate`, `search_procedure`, `search_learners`, `discover` | Case files, commands, `discover` |
| Metacognition runtime | Router, controller, recorder, verdicts, detector, learner, practice, gate, approval, executor, budget, events | Blame, routing, budget, judge, worker |
| SDK surface | `Agent`, `Runner`, tools, guardrails, sessions, streaming, tracing | None |
| helix loop | Hooks, `RequestPatch`, memory tiers, code-skill guard, learning mechanisms, GEPA | Never modified |
| Model access | LiteLLM, recording and replay, prices | Effort ladders per provider |

Phase 1's rules stand. Every capability is a hook around helix's loop, never a change to it, and every model
call can be recorded and replayed.

### 3.2 Processes

Today one Python process serves runs, records them and learns in background asyncio tasks, and the CLI reads
the same SQLite file. That suits experiments but not a deployed agent. A restart loses a learning attempt,
paid writer calls and model-written code share the serving process, and no other process can do the learning.

The target splits three roles that share one store:

- **Serving** (your application, or `metacog serve`). For each run it routes, chooses the reasoning setting,
  runs, records, judges when a verifier is set, and runs the detector. It never calls a writer model.
- **Learning** (`metacog worker`). It finds procedures that are due for an attempt, learns from the runs since
  quarantine, gates the candidate, and installs or proposes it. The schedule lives in the store, so a restart
  resumes where it stopped. A lease keeps it to one worker per store.
- **Operating** (the CLI, the service's endpoints and the dashboard): feedback, approval, quarantine,
  calibration and reports.

`learning="inline"` (for tests and replays) and `learning="background"` (one process, as today) stay;
`learning="worker"` is added.

### 3.3 One agent definition for every process

The worker and the service must build exactly the agent the application runs, so they import it by name:

```python
# app.py
meta = Metacognition([reimbursable_total], verifier=check, practice=AgentPractice(world=ExpenseWorld),
                     store="runs/jordan.meta", learning="worker")
agent = Agent(name="Expense assistant", instructions=PROMPT, tools=[read_trip, read_policy, file_claim],
              model="gpt-5.4-mini", metacognition=meta)
```

```text
metacog serve app:agent       host the agent over HTTP
metacog worker app:agent      learn for it
metacog status app:agent      or a store path, as today
```

### 3.4 Components

| Component | Job | Phase 2 change |
|---|---|---|
| Router | Decide which procedures a request belongs to, and offer the trusted ones | A calibrated threshold per procedure, a "no procedure" class, and match scores recorded |
| Controller | Choose the reasoning setting by lookup | Keyed by model. Takes the most demanding floor among the offered procedures. Optionally escalates when the agent ignores what it was offered (experimental) |
| Recorder | Store each run: verdict, effort, cost, and the procedures matched and offered | Adds the model, the procedures the run called, and match scores |
| Verdicts | Accepted or rejected, with a source | Adds a judge (a model with a rubric). A judge may inform the detector but never the gate |
| Detector | Quarantine a version whose recent uses keep failing | Blames the procedures a run called, not everything offered. The calendar parity keeps "offered" |
| Learner | Write a candidate from the runs since quarantine | Runs under a budget |
| Practice | Try a candidate on a past case: as a function, through the agent in a rebuilt world, or in a replayed world | None |
| Gate | Admit a candidate only if it clears the bar on held-back cases | None |
| Approval | Install, or propose to a person | A proposal can notify (webhook) |
| Executor | Run procedure code | Chosen by author: hand-written code runs guarded in-process, model-written code in a subprocess |
| Budget | Cap what learning spends | New: dollars and attempts per procedure per day |
| Events | Tell the outside world | Webhooks beside helix's bus |

### 3.5 The store

One SQLite database per deployment, as now, with these additions:

- WAL mode and a busy timeout, so the serving process, the worker and the CLI can share it;
- a schema version, with migrations;
- three new fields on each run: the model, the procedures it called, and its match scores;
- a floors table mapping model, procedure and arm to a setting, with the bar, case count and date measured.
  `calibrate` writes it and the controller reads it;
- the worker's lease and the learning schedule, in the state table.

A second backend such as Postgres is not planned. The store's methods are the seam if one is ever needed.

### 3.6 Effort ladders

Today the reasoning rungs are GPT-5.x names sent as `reasoning_effort`. A ladder maps the harness's rungs
(`none`, `low`, `medium`, `high`) to one model family's parameters. OpenAI's ladder comes first. Anthropic's
follows once a test pins exactly what LiteLLM sends for it; keys for both providers are in `.env`. Floors are
keyed by model, so a model that hasn't been calibrated runs at the agent's own setting until it is.

## 4. What the review found

**Agents with more than one procedure**

1. **Every request matches every procedure.** `relevance` defaults to 0.0 and the lexical vectors have no
   negative entries. So `_match` (`meta/runtime.py:240`) returns the top `recall_k` procedures for any request.
   An unrelated request gets a procedure's reasoning setting, and every run counts as learning experience for
   every procedure.
2. **Failures are blamed on everything offered.** `_observe(plan.offered, ...)` (`meta/runtime.py:334`) counts
   a rejected run against each offered procedure, whether the agent called it or not. With two procedures, one
   failing procedure can quarantine both.
3. **The effort comes from the first offered procedure** (`meta/runtime.py:299`), even when the procedures'
   floors differ.

**More than one model**

4. **The rungs are GPT-5.x names sent as one parameter** (`effort_param`, `meta/runtime.py:173`;
   `run.py:198`). Neither the floors nor the controller's table is keyed by model (`offline.py:53`).
5. **Model defaults are pinned.** The offline searches default to `gpt-5.4-mini` (`offline.py:234` and
   `:312`), and the runtime's last-resort writer ignores `METACOG_DEFAULT_MODEL` (`meta/runtime.py:414`).

**Deployment**

6. **Learning runs inside the serving process** (`meta/runtime.py:386`). It also needs `meta.agent`, which only
   an online run sets, so no other process can learn.
7. **The store has no WAL mode, busy timeout or schema version** (`meta/store.py:57`). A second process risks
   "database is locked", and a stored deployment has no upgrade path.
8. **Model-written code runs in-process by default** (`executor="guarded"`, `meta/runtime.py:176`). helix calls
   its guard "not a sandbox", and the plan (decision 10) said model-written code would run in a subprocess by
   default.
9. **Learning has no spending limit.** A procedure that never passes the gate is retried every
   `relearn_after` runs, indefinitely.
10. **Floors live in loose JSON files**, and the controller is assembled by hand
    (`examples/expenses/study.py:128`).
11. **There is no judge.** `Verdict` names the source (`meta/parts.py:33`) but nothing produces it, so an agent
    without a programmatic checker can only learn from people.
12. **Background learning writes to a report that was already returned** (`meta/runtime.py:386` and `:436`).
13. **Recordings get Windows line endings** (`recording.py:175`).

**Platform**

14. **Evaluation is Python only.** Cases live in code, and there are no case files and no `metacog calibrate`.
15. **Nothing hosts an agent.** There is no HTTP API for runs, feedback or approvals.
16. **There is no live view.** `metacog report --html` is a snapshot.
17. **It isn't installable.** helix comes from a linked checkout (`scripts/link_helix.py`); decision 1 of the
    plan is still open.
18. **There is no CI.** It waits on the push, and the parity tests need the book repo at a pinned commit.

## 5. Getting it in place

Every milestone keeps the parity suite green and ends with tests that need no API key.

**Progress.** A1 was built on 2026-11-04 (`tests/test_a1.py`, 18 tests; 89 in all):

- Findings 1 to 5, 12 and 13 are fixed. WAL mode and the busy timeout (part of finding 7) came with the store
  migration.
- The E5 replay still reproduces the study's trace under the new default of blaming only the procedures a
  run called: both rejected runs that were offered the stale procedure had called it.
- LiteLLM (1.106, and 1.108 as pinned) sends `reasoning_effort` to Claude as no thinking for `none`, and
  otherwise as adaptive thinking at that effort (Claude 5) or a thinking budget (earlier models). A test pins
  this.

A2 was built the same day (`tests/test_a2.py`, 11 tests; 100 in all):

- Findings 6 to 11 are fixed: `learning="worker"` and `metacog worker`, the shared store, `executor="auto"`,
  `LearningBudget`, `Judge`, and floors in the store.
- With `executor="auto"`, the E5 replay now runs its learned procedure in a child process (16 sandboxed
  calls) and still sends byte-identical requests.
- The review missed a race, which is now fixed: two serving processes could give two runs the same number.
  Numbering a run or a version now holds SQLite's write lock from read to write.
- Not yet: a worker for `PerUser` deployments (one store per person).

A3 was built next (`tests/test_a3.py`, 11 tests; 111 in all), with the dashboard kept lean as agreed:

- Findings 14 to 16 are fixed:
  - **Evaluation:** case files and `metacog calibrate`, which records floors per model and, from labeled
    cases, a calibrated router in the store; a new process on the store picks both up.
  - **Hosting:** `metacog serve`.
  - **Live view:** the dashboard, with what needs a person, a run feed with each run's detail, and procedure
    states. Analytics stay in the static report.
- **`metacog discover` finds work worth a procedure.** It groups runs without a procedure by the tools they
  used and ranks the groups by reasoning spent. It prints a skeleton stub per group, or with `--draft` a writer
  model drafts one.
- **Webhooks** post quarantines, proposals and installs from whichever process causes them.
- **`examples/calendar/drift_demo.py`** replays chapter 6's E5 through run 12, with a person approving
  installs. The dashboard then has a real quarantine and a real proposal to show with no API key. The
  acceptance tests serve the expense agent over HTTP from its recorded calls, judge its runs through
  feedback, and approve the demo's proposal through the API.
- **Two bugs turned up only when the real server and page ran,** and both are fixed: the summary's learning
  totals overwrote the deployment's learning mode, and run details printed placeholder text.
- **Routing on a description alone is weak.** The live demo showed run 8's request scoring 0.00 against its
  procedure, because the two share no words. The study's single procedure matched only because nothing was
  thresholded. Example requests (from labeled cases) are what make routing work.

**A1. More than one procedure, more than one model** (findings 1 to 5, 12 and 13)

- Record each run's model, the procedures it called, and its match scores. Blame verdicts on called
  procedures, keeping "offered" as an option for the calendar parity.
- Calibrate a routing threshold per procedure from labeled cases, and add the "no procedure" class.
- Add effort ladders (OpenAI, then Anthropic) and key the floors by model. The controller takes the most
  demanding floor among the offered procedures.
- Default the writer and proposer to the agent's model.
- Send background learning events to the store and bus, not to the returned report. Write recordings with LF.
- Add a schema version and migrations to the store, since the runs table changes.
- *Accept:* in a scripted agent with two procedures, failures of one quarantine only that one. Unrelated
  requests run at the agent's own setting. Ladder tests pin the request parameters LiteLLM sends for both
  providers. A store written before A1 opens and migrates.

**A2. Deployment** (findings 6 to 11)

- `learning="worker"` and `metacog worker`, with a lease and the schedule in the store.
- The store gets WAL mode, a busy timeout and the floors table.
- The executor is chosen by author. Add the learning budget and `Judge`.
- *Accept:* a worker killed mid-attempt resumes. Serving, worker and CLI write at once without lock errors.
  Hostile learned code is contained by default. The budget stops learning and records why.

**A3. Platform** (findings 14 to 16)

- Case files (JSONL), and `metacog calibrate`, which writes floors per model into the store.
- `metacog serve`, as an optional `serve` extra: runs, feedback, procedures, approve, quarantine, report and
  webhooks.
- A dashboard: runs and their steps, verdicts, procedure versions with source and lineage, proposals with an
  approve button, and the economics.
- `metacog discover`: group recorded runs by the tools they used, rank the groups by reasoning spent, and
  draft a procedure stub for the top ones.
- *Accept:* replaying recorded calls, the expense agent is served, judged through feedback over HTTP, and has
  a proposal approved from the dashboard.

**A4. Distribution** (findings 17 and 18)

- Packaging under decision 1, CI on GitHub after the push, and a guide that takes the expense agent through
  every stage in section 2.
- *Built:*
  - **Packaging:** the side project sits in the book repository (decision 1).
  - **CI:** `.github/workflows/metacognition.yml` runs lint and the suite whenever `metacognition/`, `helix/`
    or `chapters/ch06/` changes, with the versions pinned in `constraints.txt`.
  - **The guide:** [GUIDE.md](GUIDE.md) walks the expense agent from design to drift.
  - **New commands:** `examples/expenses/app.py` makes the expense agent a deployment the CLI can name,
    `metacog install` trusts a hand-written procedure, and `metacog serve --recording` replays or records a
    served agent's calls.
  - **Tests:** the guide's free steps run through the CLI in `tests/test_a4.py` and reproduce the smoke
    study's numbers.

## 6. Example domains

Protocols and results live in [DOMAINS.md](DOMAINS.md). D1 ran live on 2026-11-05 for $1.05 and repeated the
study's pattern on the expense domain. The domain-neutral prompts learned a procedure that matches the
hand-written one on all 16 validation trips. In the online loop the procedure was quarantined after run 8 and
relearned at run 12 (gate 6/6), and learning paid for itself after 8.5 requests.

Each domain is built with the harness the way a real agent would be: tools, procedures, a verdict source,
calibration, deployment and then drift. Each answers one question about metacognition. Live runs stop at a
set budget.

| | Domain | The question | What it exercises |
|---|---|---|---|
| D1 | Expense claims (built) | Does the study's result hold on a second domain, with repeated trials and the domain-neutral prompts? | The full M5 study: floors, learning, and drift when the exchange-rate month changes |
| D2 | Store refunds | Can an agent learn safely when verdicts are scarce, late and human? | Simulated reviewers who judge a sample of runs a day later; a judge as a second signal; replayed practice; approval by a person; a holiday returns policy as drift |
| D3 | Invoice intake | How does metacognition behave with several procedures and several customers? | A procedure per task and per-customer scopes; routing and blame; `discover`; a vendor changing its layout |
| D4 | Model switch | Do floors and learned procedures carry over between models? | D1 and D3 on `gpt-5.4-mini` and a Claude model: ladders, floors per model, and a procedure written by one model used by another |

I estimate $10 to $20 per domain on `gpt-5.4-mini`, and more for D4's Claude runs.

## 7. Out of scope

Handoffs, token streaming, parallel tool calls, the Responses API, MCP servers (until a domain needs one), and
stores shared across hosts.

## 8. Decisions needed

1. **How helix is packaged** (decision 1 of the plan). *Settled on 2026-11-04:* the harness lives in the book
   repository as a side project (`metacognition/`). helix stays unpackaged, and `scripts/link_helix.py` links
   the repository root into the project's environment, so the side project always runs against the helix
   beside it. The standalone repository it grew in (milestones M0 to A2) is its history.
2. **The service and dashboard.** I recommend building them (A3) as an optional extra, because deploying is
   part of the goal.
3. **The domains.** I recommend D1 to D4 in that order; swap in any domain you'd rather explore.
4. **The push.** The side project reaches GitHub with the book repository; CI (A4) can then run its tests.
