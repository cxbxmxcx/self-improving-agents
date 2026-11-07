"""Jordan's expense assistant as a deployment that `metacog` commands can name (GUIDE.md walks through it).

    python -m examples.expenses.app cases dev 4 examples/runs/expense_dev.jsonl     # write a case file
    metacog install examples.expenses.app:agent reimbursable_total examples/expenses/memories/reimbursable_total.py
    metacog calibrate examples.expenses.app:agent examples/runs/expense_dev.jsonl --world examples.expenses.app:world
    metacog serve examples.expenses.app:agent --context examples.expenses.app:context
    metacog worker examples.expenses.app:agent

The store is examples/runs/jordan.meta. Learning happens in the worker, and an
install follows the gate because the finance team's checker is code (`check`).
Practice runs a candidate on a case's own world, as chapter 6's act does, and
the gate tries it on the trips the study held back after the policy change.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from examples.expenses.agent import assistant, check, practice, reimbursable_total
from examples.expenses.world import ExpenseWorld, task_by_id, task_set
from metacog import Agent
from metacog.meta import Gate, Metacognition
from metacog.offline import Case, save_cases

STORE = Path(__file__).resolve().parents[1] / "runs" / "jordan.meta"


def build(store: str | Path = STORE, **options: Any) -> Agent:
    """The deployment on `store`; `options` override the Metacognition settings."""
    settings: dict[str, Any] = dict(verifier=check, practice=practice, recall_k=1, relearn_after=4,
                                    learning="worker", gate=Gate(cases=[t["id"] for t in task_set("gate")]))
    settings.update(options)
    return assistant(metacognition=Metacognition([reimbursable_total], store=str(store), **settings))


def agent() -> Agent:
    """The deployment `examples.expenses.app:agent` names."""
    return build()


def world(data: dict[str, Any]) -> ExpenseWorld:
    """A case's context, rebuilt from its data (the task: trip, receipts and policy)."""
    return ExpenseWorld(data)


def context(body: dict[str, Any]) -> ExpenseWorld:
    """A served run's context, from the case_id the request names (a trip such as E001)."""
    return ExpenseWorld(task_by_id(body["case_id"]))


def write_cases(path: str | Path, split: str = "dev", n: int = 16) -> Path:
    """A case file of the first `n` trips of a split (dev, validation, test, stream, gate)."""
    return save_cases(path, [Case(t["id"], t["text"], t) for t in task_set(split)[:n]])


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[0] != "cases":
        print("usage: python -m examples.expenses.app cases SPLIT N PATH", file=sys.stderr)
        return 2
    _, split, n, path = argv
    print(f"wrote {write_cases(path, split, int(n))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
