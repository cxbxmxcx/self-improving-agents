"""The learning worker: learn for a deployment from its store, off the serving path.

    metacog worker app:agent            run passes forever, one every 30 seconds
    metacog worker app:agent --once     one pass (for cron, or a test)

The target names a metacog Agent whose metacognition has `learning="worker"`:
`module:attribute`, where the attribute is the Agent or a function returning
one. Serving processes record runs and quarantine procedures; the worker finds
attempts that are due and runs them through the same agent definition.

A worker holds the store's "worker" lease while it works and renews it during
long attempts, so two workers never learn for one store at once. A worker that
dies stops renewing; once its lease lapses the next worker takes over and
retries whatever attempt it left unfinished (`Metacognition.learn_due`).
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from helix.metacognition.exceptions import UserError

LEASE = "worker"


def load_object(target: str) -> Any:
    """What a `module:attribute` target names, imported with the current directory on the path."""
    module_name, _, attribute = target.partition(":")
    if not module_name or not attribute:
        raise UserError(f"a target is module:attribute, such as app:agent, not {target!r}")
    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        return getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise UserError(f"can't load {target}: {exc}") from exc


def load_agent(target: str) -> Any:
    """The metacog Agent a `module:attribute` target names (calling the attribute if it is a factory)."""
    def is_agent(value: Any) -> bool:
        return hasattr(value, "metacognition") and hasattr(value, "tools")

    found = load_object(target)
    if not is_agent(found) and callable(found):
        found = found()
    if not is_agent(found):
        raise UserError(f"{target} is not a metacog Agent")
    return found


def holder_name() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


async def work(agent: Any, *, once: bool = False, interval: float = 30.0, lease: float = 300.0,
               holder: str | None = None, out: Callable[[str], Any] = print) -> int:
    """Run learning passes for `agent` until cancelled (or one, with `once`); return the passes made."""
    meta = getattr(agent, "metacognition", None)
    if meta is None or not hasattr(meta, "learn_due"):
        raise UserError("the worker needs an agent with Metacognition (one store; PerUser isn't supported yet)")
    meta.agent = agent  # AgentPractice runs candidates through it
    holder = holder or holder_name()
    store = meta.store
    passes, waiting = 0, False

    async def renew() -> None:
        while True:
            await asyncio.sleep(lease / 3)
            store.acquire_lease(LEASE, holder, lease)

    try:
        while True:
            if store.acquire_lease(LEASE, holder, lease):
                waiting = False
                renewal = asyncio.create_task(renew())
                try:
                    for event in await meta.learn_due(holder=holder):
                        out(event)
                finally:
                    renewal.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await renewal
            elif not waiting:
                waiting = True
                current = store.lease(LEASE) or {}
                out(f"waiting: another worker ({current.get('holder', '?')}) holds this store's lease")
            passes += 1
            if once:
                return passes
            await asyncio.sleep(interval)
    finally:
        store.release_lease(LEASE, holder)
