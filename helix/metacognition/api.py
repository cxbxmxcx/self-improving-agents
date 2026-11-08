"""The `api` object a learned procedure receives: read-only tools as plain functions.

Learned code is synchronous and runs in a worker thread under helix's
code-skill guard. Each tool listed in a procedure's `api=[...]` appears here
as a synchronous function: a sync tool runs in that worker thread, and an
async tool is scheduled on the serving event loop and waited for. The tool
gets this run's context, as it would if the model had called it.

Only read-only tools are offered (chapter 6's rule: procedures compute, the
agent acts), and the guard keeps learned code away from this object's
private attributes, which all begin with an underscore.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

from helix.metacognition.run_context import RunContextWrapper
from helix.metacognition.tool import FunctionTool

_ON_LOOP = "an async api tool was called on the event loop's own thread, where waiting for it would deadlock"


async def _awaited(awaitable: Any) -> Any:
    return await awaitable


class SkillAPI:
    def __init__(self, tools: list[FunctionTool], ctx: RunContextWrapper[Any], loop: asyncio.AbstractEventLoop,
                 timeout: float = 30.0, executor: Any = None) -> None:
        self._tools = {tool.name: tool for tool in tools if tool.read_only}
        self._ctx = ctx
        self._loop = loop
        self._timeout = timeout
        # A run's tool executor (a replayed world) answers the procedure's calls too.
        self._executor = executor

    def __getattr__(self, name: str) -> Any:
        tools = self.__dict__.get("_tools", {})
        if name not in tools:
            raise AttributeError(f"api has no {name!r}; it offers {sorted(tools)}")
        tool: FunctionTool = tools[name]

        def call(*args: Any, **kwargs: Any) -> Any:
            names = list(tool.params_model.model_fields)
            if len(args) > len(names):
                raise TypeError(f"{name}() takes {len(names)} arguments but {len(args)} were given")
            arguments = {**dict(zip(names, args, strict=False)), **kwargs}
            if (self._executor is not None or inspect.iscoroutinefunction(tool.fn)) and self._on_loop_thread():
                raise RuntimeError(_ON_LOOP)
            if self._executor is not None:
                result = self._executor(tool, self._ctx, arguments)
            else:
                result = tool.fn(self._ctx, **arguments) if tool.takes_context else tool.fn(**arguments)
            if not inspect.isawaitable(result):
                return result
            if self._on_loop_thread():
                getattr(result, "close", lambda: None)()
                raise RuntimeError(_ON_LOOP)
            return asyncio.run_coroutine_threadsafe(_awaited(result), self._loop).result(self._timeout)

        return call

    def _on_loop_thread(self) -> bool:
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    def __dir__(self) -> list[str]:
        return sorted(self._tools)
