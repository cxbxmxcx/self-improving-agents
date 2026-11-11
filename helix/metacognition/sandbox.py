"""Run learned code in a separate process: metacog's `executor="subprocess"`.

helix's code-skill guard (an AST check, a builtin allow-list, a range cap, a
deadline) contains honest mistakes, and helix says plainly that it is not a
security sandbox. Model-written procedures in production run here instead:

- each call starts a fresh `python -I` child that imports only the standard
  library, so it holds no agent state, keys, or connections;
- the child runs the function with the same builtins and range cap, and asks
  this process over a pipe whenever the code calls `api.<tool>(...)`; the
  read-only tool runs here, with the run's context, and its result goes back;
- the source still passes helix's guard first, and a timer kills the child if
  it outlives its deadline plus a startup allowance.

Arguments and results cross the pipe as JSON, so tuples come back as lists.
A process start costs tens of milliseconds, which is the price of isolation.
OS-level limits (memory, network, files) belong to the container or job the
agent runs in; this layer only keeps learned code out of the agent's process.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from helix.memory.code_skills import SAFE_BUILTINS, args_model
from helix.tools import Tool

CHILD = Path(__file__).with_name("sandbox_child.py")
STARTUP_ALLOWANCE = 10.0


def run_sandboxed(name: str, source: str, kwargs: dict[str, Any], api_call: Callable[[str, list, dict], Any],
                  api_names: list[str], timeout: float, python: str = sys.executable) -> Any:
    """Run one call of a learned function in a child process. Blocking; call it from a worker thread."""
    proc = subprocess.Popen([python, "-I", str(CHILD)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8")
    killed = threading.Event()

    def kill() -> None:
        killed.set()
        proc.kill()

    timer = threading.Timer(timeout + STARTUP_ALLOWANCE, kill)
    timer.start()

    def send(message: dict[str, Any]) -> None:
        proc.stdin.write(json.dumps(message, default=str) + "\n")
        proc.stdin.flush()

    try:
        send({"name": name, "source": source, "kwargs": kwargs, "api": api_names,
              "builtins": sorted(SAFE_BUILTINS), "timeout": timeout})
        while True:
            line = proc.stdout.readline()
            if not line:
                if killed.is_set():
                    return {"error": f"{name} was stopped after {timeout:g} seconds"}
                return {"error": f"{name} crashed: {proc.stderr.read().strip()[-500:] or 'no output'}"}
            message = json.loads(line)
            if "call" in message:
                try:
                    send({"result": api_call(message["call"], message.get("args", []), message.get("kwargs", {}))})
                except Exception as exc:  # noqa: BLE001 - the tool's error goes back to the learned code
                    send({"error": f"{type(exc).__name__}: {exc}"})
            elif "done" in message:
                return message["done"]
            else:
                return {"error": message.get("failed", f"{name} failed")}
    except (BrokenPipeError, OSError) as exc:
        return {"error": f"{name} could not run in the sandbox: {exc}"}
    finally:
        timer.cancel()
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            stream.close()


def sandboxed_tool(name: str, description: str, source: str, api: Any, *, timeout: float = 5.0) -> Tool:
    """A helix Tool that runs a learned function in a child process, with `api` served from this one.

    The function is compiled here only to read its signature: the guard has
    already refused imports and code that runs on load, so compiling defines
    the function without running its body.
    """
    model = args_model(source, name, ("api",))
    api_names = sorted(dir(api))

    def api_call(tool: str, args: list, kwargs: dict) -> Any:
        return getattr(api, tool)(*args, **kwargs)

    async def call(**kwargs: Any) -> Any:
        return await asyncio.to_thread(run_sandboxed, name, source, kwargs, api_call, api_names, timeout)

    return Tool(name=name, description=description, args_model=model, fn=call)
