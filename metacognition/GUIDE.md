# Building an agent that learns to think less

This guide takes the expense assistant in `examples/expenses` through every stage of an agent's life, the
stages in [ARCHITECTURE.md](ARCHITECTURE.md) §2, with the command for each. Steps marked **free** replay
recorded model calls. Steps marked **paid** make live calls with keys from a `.env` file in this folder.

Jordan files travel expense claims. The agent reads a trip and the company's expense policy, works out the
reimbursable amount in US dollars, and files one claim, and the finance team's checker accepts or rejects it.
The rules that matter (per-diem caps, the company's exchange rates, notes about which receipts count) are in
the policy, not in the request, so this is work where more reasoning helps only so much and a procedure helps
a lot.

## 0. Set up

```powershell
cd metacognition
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"
.venv\Scripts\python scripts\link_helix.py
.venv\Scripts\python -m examples.expenses.app cases dev 4 examples/runs/expense_dev.jsonl    # free
```

The last line writes a case file of four development trips. Each line holds a trip's id, its request, and the
trip, receipts and policy that rebuild its world.

## 1. Design: declare the agent, the work worth a procedure, and how runs are judged

[`examples/expenses/agent.py`](examples/expenses/agent.py) is an ordinary agent with three tools.
`read_trip` and `read_policy` are marked `read_only=True`, so a learned procedure may call them. `file_claim`
acts, so only the agent calls it. The procedure is a typed stub with no body:

```python
@procedure(api=[read_trip, read_policy])
def reimbursable_total(trip_id: str) -> dict:
    """Work out how much of a trip's receipts the company reimburses, in US dollars, under its expense policy.

    Returns: {"total_usd": float}, the reimbursable total rounded to the cent.
    """
    ...
```

Runs are judged by `check`, the finance team's checker. [`examples/expenses/app.py`](examples/expenses/app.py)
turns the agent into a deployment: a store, learning in a worker, and a gate on trips held back from learning.
Every command below names it as `examples.expenses.app:agent`.

Not sure which work deserves a procedure? Serve the agent without one for a while, then run
`metacog discover examples.expenses.app:agent`. It groups the recorded runs by the tools they used and ranks
the groups by the reasoning they spent, which is what a procedure saves.

## 2. Evaluate: how much reasoning does the work need?

```powershell
metacog calibrate examples.expenses.app:agent examples/runs/expense_dev.jsonl --world examples.expenses.app:world `
  --efforts none high --max-turns 12 --recording examples/expenses/recorded/smoke.jsonl --mode replay   # free
```

With no procedure trusted yet, only the no-memory arm is measured. The agent solves 0 of 4 trips at `none` and
2 of 4 at `high`, so no setting clears the bar and the floor falls back to the best one, `high`. The floors
are recorded in the store for `gpt-5.4-mini`, and the controller reads them from there. Drop `--recording` and
`--mode` to measure all four settings live (**paid**).

## 3. Bootstrap: where the first procedure comes from

The procedure can come from a person, from the agent's own runs, or from a search.

- **By hand (free):** `metacog install examples.expenses.app:agent reimbursable_total examples/expenses/memories/reimbursable_total.py`.
  It must pass the code-skill guard and take `api` first. Code a person wrote runs in-process.
- **Learned online (paid):** serve the agent with no procedure. Once runs have been judged, the worker writes
  one from them (step 6 shows the worker).
- **Searched offline (paid):** `metacog.offline.search_procedure` runs GEPA over a seed version's code, as the
  study's E2 did.

## 4. Release: what to trust

Run the step 2 command again. Now that the procedure is trusted, the memory arm is measured too, and the agent
with the procedure solves 4 of 4 at `none` as well as at `high`. The recorded table becomes `{'memory': 'none',
'no_memory': 'high'}`: a request the procedure covers runs at `none`, anything else at `high`.

Versions a model writes earn trust differently. The gate tries each candidate on the held-back trips, and it
runs in a child process rather than in the agent's. With a programmatic checker like `check`, a version that
passes is installed. With people or a `Judge` as the verifier, it waits for approval in the dashboard or from
`metacog approve`.

## 5. Operate: serve it, watch it, judge it

```powershell
metacog serve examples.expenses.app:agent --context examples.expenses.app:context    # paid per run
```

`--context` builds each run's world from the request's `case_id`. Post a run with
`{"input": "...", "case_id": "E001"}` to `/api/runs`, sending the header `X-Metacog: 1`. Add
`--recording examples/runs/served.jsonl` to save every live reply so the runs replay later for free.

At http://127.0.0.1:8620 the dashboard shows each run's routing, setting, procedure calls and verdict as it
lands, and what needs a person. When the checker isn't code, people judge runs there or with
`metacog feedback examples.expenses.app:agent RUN accept|reject --reason "..."`. `metacog status`, `runs` and
`report` give the same picture in the terminal. To see the dashboard on real history with no key, run
chapter 6's drift demo (the README's dashboard section).

## 6. Adapt: when the world changes

In this world, the policy changes: receipts start converting at their own month's rates, not the month the trip
started. The stale procedure's claims are rejected and the reason names the rates. After two rejections in its
last four uses, the detector quarantines it. Runs fall back to the no-memory setting, and the worker relearns
from the runs since:

```powershell
metacog worker examples.expenses.app:agent --webhook https://hooks.example/metacog    # paid: writer calls
```

The worker holds a lease on the store, so it learns alone. Each due attempt learns from the judged runs since
the quarantine and gates the candidate on the held-back trips. A pass installs the candidate, which is the
default here because `check` is code, and the webhook posts each quarantine and install. Cap the spend with
`LearningBudget` in `app.py`. To run the whole loop through the policy change in one process (**paid**; `--budget`
stops it at a dollar limit), use
`python -m examples.expenses.study --phases online --memory reference --mode record --budget 5`.

## 7. Revisit: when the model changes

Floors belong to a model. Point the agent at another model, for example `claude-haiku-5-5`, and until it is
calibrated, its runs use the agent's own setting, not floors measured on `gpt-5.4-mini`. Run step 2's
command (**paid**) to record floors for the new model beside the old ones. Anthropic models get their reasoning
settings through LiteLLM, as [`metacog.ladders`](src/metacog/ladders.py) describes.
