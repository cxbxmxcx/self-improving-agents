"""Find recurring, costly work in a deployment's recorded runs: candidates for a new procedure.

This is metacognition at design time. Before you can declare a procedure you
have to notice the work worth one, and the recorder has already seen it: runs
that reach for the same tools again and again, and spend reasoning each time.

    metacog discover app:agent             the groups, ranked, each with a skeleton stub
    metacog discover app:agent --draft     a writer model fills in each stub from example runs

Runs are grouped by the set of tools they used, leaving out procedure calls;
runs that called a procedure are already covered and are skipped. Groups are
ranked by the reasoning tokens they spent, since that is what a procedure saves.
For each group you get how often it recurs, what it cost, how often it was
accepted, example requests, and a skeleton `@procedure` stub whose `api` lists
the group's read-only tools. A stub is a starting point for a person to finish:
its name, signature and `Returns:` line are what metacognition learns against.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

DRAFT_PROMPT = """An AI agent keeps doing the same kind of work, reasoning it out each time. Below are
runs of that work: each request, the tools the agent called with what they returned, and its
final answer. The work is worth turning into one Python function the agent can call instead.

Write a typed stub for that function, in this form and nothing else:

```python
@procedure(api=[{api}])
def name(argument: type, ...) -> dict:
    \"\"\"What the function computes, in one or two sentences a model can act on.

    Returns: the shape of the result, such as {{"total_usd": float}}.
    \"\"\"
    ...
```

Take the parameters from what varies between the requests (such as ids), not from what the
tools return. The function may call only the read-only tools named in `api`, and does not act:
the agent still takes any action itself. Keep the body as `...`."""


@dataclass
class WorkGroup:
    """Runs that used the same tools: one kind of recurring work."""

    tools: tuple[str, ...]
    runs: int = 0
    judged: int = 0
    accepted: int = 0
    reasoning_tokens: int = 0
    dollars: float = 0.0
    requests: list[str] = field(default_factory=list)
    run_ids: list[str] = field(default_factory=list)

    @property
    def mean_reasoning_tokens(self) -> float:
        return self.reasoning_tokens / self.runs if self.runs else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"tools": list(self.tools), "runs": self.runs, "judged": self.judged, "accepted": self.accepted,
                "reasoning_tokens": self.reasoning_tokens, "mean_reasoning_tokens": self.mean_reasoning_tokens,
                "dollars": self.dollars, "requests": self.requests[:3]}


def discover(store: Any, *, procedures: Iterable[str] = (), min_runs: int = 2) -> list[WorkGroup]:
    """Group a store's runs by the tools they used and rank the groups by reasoning spent."""
    runs = store.runs()
    known = set(procedures) | {item.split("@")[0] for run in runs for item in run["offered"] or []}
    groups: dict[tuple[str, ...], WorkGroup] = {}
    for run in runs:
        used = {step["tool"] for step in (run["content"] or {}).get("steps") or []}
        if run.get("called") or used & known:
            continue  # a procedure already does this work
        tools = tuple(sorted(used))
        if not tools:
            continue
        group = groups.setdefault(tools, WorkGroup(tools))
        group.runs += 1
        group.judged += run["verdict"] is not None
        group.accepted += run["verdict"] == "accepted"
        group.reasoning_tokens += run["reasoning_tokens"] or 0
        group.dollars += run["dollars"] or 0.0
        group.run_ids.append(run["run_id"])
        if run["request"] not in group.requests:
            group.requests.append(run["request"])
    ranked = [g for g in groups.values() if g.runs >= min_runs]
    return sorted(ranked, key=lambda g: (-g.reasoning_tokens, -g.runs, g.tools))


def skeleton(group: WorkGroup, read_only: Iterable[str] | None = None) -> str:
    """A stub to start from: `api` holds the group's read-only tools (all of its tools when unknown)."""
    allowed = set(group.tools if read_only is None else read_only)
    api = ", ".join(tool for tool in group.tools if tool in allowed)
    return (f"@procedure(api=[{api}])\n"
            "def name_this(...) -> dict:\n"
            '    """What this work computes, in one or two sentences a model can act on.\n\n'
            "    Returns: the shape of the result.\n"
            '    """\n'
            "    ...\n")


def _render(run: dict[str, Any], limit: int = 600) -> str:
    content = run["content"] or {}
    lines = [f"Request: {run['request']}"]
    for step in content.get("steps") or []:
        result = json.dumps(step.get("result"), default=str)
        lines.append(f"- {step['tool']}({json.dumps(step.get('arguments') or {})}) -> "
                     f"{result if len(result) <= limit else result[:limit] + ' ...'}")
    answer = content.get("answer")
    if answer is not None:
        lines.append(f"Final answer: {str(answer)[:limit]}")
    return "\n".join(lines)


async def draft_stub(group: WorkGroup, store: Any, *, model: str, read_only: Iterable[str] | None = None,
                     examples: int = 3, params: dict[str, Any] | None = None) -> str:
    """Ask a writer model to fill in the group's stub from example runs. One model call."""
    from helix import llm_call

    allowed = set(group.tools if read_only is None else read_only)
    api = ", ".join(tool for tool in group.tools if tool in allowed)
    runs = [run for run in (store.run(run_id) for run_id in group.run_ids[:examples]) if run is not None]
    messages = [{"role": "system", "content": DRAFT_PROMPT.format(api=api)},
                {"role": "user", "content": "\n\n".join(_render(run) for run in runs)}]
    response = await llm_call.acompletion(model=model, messages=messages, **(params or {}))
    reply = response.choices[0].message.content or ""
    found = re.search(r"```(?:python)?\s*\n(.*?)```", reply, re.DOTALL)
    return (found.group(1) if found else reply).strip() + "\n"
