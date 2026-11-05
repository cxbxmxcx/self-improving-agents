"""Practice without a simulator: replay an accepted run's world, with side effects dry-run.

AgentPractice needs a factory that rebuilds a case's world, and many
deployments have none: the calendar is real, the order system is live.
ReplayPractice rebuilds the world from the accepted run itself:

- read-only tools answer from what the run recorded, matched on their full
  arguments (defaults filled in); with `live_context`, a read the run never
  made goes to the real tool;
- side-effecting tools never run: the call is noted and answered as the
  recorded run was answered, or with {"ok": true, "dry_run": true};
- the candidate passes when its actions match the accepted run's actions
  (ignoring arguments listed in `ignore`, such as a free-text title), or, when
  the accepted run took no actions, when its final answer matches.

A replayed world only knows what the run looked at, so a candidate that
reads something else sees an error rather than the world: practice by replay
is a conservative check, best for procedures that read what the agent read.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from metacog.meta.parts import Practice
from metacog.run_context import RunContextWrapper
from metacog.tool import FunctionTool


def _key(tool: FunctionTool, arguments: dict[str, Any], ignore: list[str]) -> tuple[str, str]:
    try:
        full = tool.params_model.model_validate(arguments).model_dump(mode="json")
    except ValueError:
        full = dict(arguments)
    return tool.name, json.dumps({k: v for k, v in full.items() if k not in ignore}, sort_keys=True, default=str)


class RecordedWorld:
    """The world of one recorded run, served to a practice run through RunConfig(tool_executor=...)."""

    def __init__(self, steps: list[dict[str, Any]], tools: dict[str, FunctionTool], *,
                 ignore: dict[str, list[str]] | None = None, live_context: Any = None) -> None:
        self.tools = tools
        self.ignore = ignore or {}
        self.context = live_context
        self.answers: dict[tuple[str, str], Any] = {}
        self.expected: list[tuple[str, str]] = []
        for step in steps:
            tool = tools.get(step.get("tool", ""))
            if tool is None:
                continue  # a procedure's own call, or a tool the agent no longer has
            self.answers.setdefault(_key(tool, step.get("arguments") or {}, []), step.get("result"))
            if not tool.read_only:
                self.expected.append(_key(tool, step.get("arguments") or {}, self.ignore.get(tool.name, [])))
        self.actions: list[tuple[str, str]] = []
        self.misses: list[tuple[str, str]] = []

    async def execute(self, tool: FunctionTool, ctx: Any, arguments: dict[str, Any]) -> Any:
        if tool.read_only:
            key = _key(tool, arguments, [])
            if key in self.answers:
                return self.answers[key]
            if self.context is not None:
                return await tool.invoke(RunContextWrapper(self.context), **arguments)
            self.misses.append(key)
            return {"error": f"{tool.name} with {key[1]} was not part of the recorded run"}
        self.actions.append(_key(tool, arguments, self.ignore.get(tool.name, [])))
        return self.answers.get(_key(tool, arguments, []), {"ok": True, "dry_run": True})

    def observed(self) -> str:
        lines = [f"actions: {self.actions or 'none'}", f"the accepted run's actions: {self.expected or 'none'}"]
        if self.misses:
            lines.append(f"reads the recorded run never made: {self.misses}")
        return "\n".join(lines)


def _same_answer(a: Any, b: Any) -> bool:
    return " ".join(str(a).split()).casefold() == " ".join(str(b).split()).casefold()


@dataclass
class ReplayPractice:
    """Practice through the agent in a recorded world. See the module docstring.

    compare   "auto" (actions when the accepted run took any, else the answer), "actions", "output",
              or a function (accepted_run, result, world) -> (passed, reason)
    ignore    per side-effecting tool, arguments left out of the comparison: {"book_time": ["title"]}
    live_context
              a factory (case_data) -> context; reads the recording lacks then go to the real tools
    """

    compare: str | Callable[[dict[str, Any], Any, RecordedWorld], tuple[bool, str]] = "auto"
    ignore: dict[str, list[str]] = field(default_factory=dict)
    live_context: Callable[[Any], Any] | None = None
    max_turns: int = 10

    def bind(self, meta: Any, name: str) -> Practice:
        async def practice(source: str, case_id: str) -> tuple[float, str, str]:
            from metacog.exceptions import AgentsException
            from metacog.run import RunConfig, Runner

            run = meta.store.accepted_run(case_id)
            if run is None:
                return 0.0, f"no accepted run of {case_id} to replay", ""
            tools = {t.name: t for t in meta.agent.tools if isinstance(t, FunctionTool)}
            live = self.live_context(run["case_data"]) if self.live_context is not None else None
            world = RecordedWorld(run["content"].get("steps", []), tools, ignore=self.ignore, live_context=live)
            config = RunConfig(metacognition_mode="practice", practice_candidate=(name, source), case_id=case_id,
                               case_data=run["case_data"], tool_executor=world.execute)
            try:
                result = await Runner.run(meta.agent, run["request"], context=live, max_turns=self.max_turns,
                                          run_config=config)
            except AgentsException as exc:
                return 0.0, f"the replayed run failed: {exc}", world.observed()
            passed, reason = self._judge(run, result, world)
            return (1.0 if passed else 0.0), reason, world.observed()

        return practice

    def _judge(self, run: dict[str, Any], result: Any, world: RecordedWorld) -> tuple[bool, str]:
        if callable(self.compare):
            return self.compare(run, result, world)
        mode = self.compare if self.compare != "auto" else ("actions" if world.expected else "output")
        if mode == "actions":
            if sorted(world.actions) == sorted(world.expected):
                return True, "the actions match the accepted run"
            return False, f"took {world.actions or 'no actions'} where the accepted run took {world.expected}"
        answer = run["content"].get("answer")
        if answer is not None and _same_answer(result.final_output, answer):
            return True, "the answer matches the accepted run"
        return False, f"answered {str(result.final_output)[:200]!r} where the accepted run answered {answer!r}"
