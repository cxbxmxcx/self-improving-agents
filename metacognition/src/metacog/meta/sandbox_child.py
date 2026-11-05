"""The child side of metacog's subprocess sandbox. Standard library only; run as a script under `python -I`.

It imports nothing from metacog or helix, so a learned function runs in a
process that holds no agent state, no API keys loaded by the parent, and no
open connections. One JSON object per line on stdin and stdout:

  parent -> child   {"name", "source", "kwargs", "api": [tool names], "builtins": [names], "timeout": seconds}
  child -> parent   {"call": tool, "args": [...], "kwargs": {...}}   the parent answers {"result": v} or {"error": m}
                    {"done": value}   or   {"failed": message}
"""

import builtins
import json
import sys
import time

MAX_RANGE = 1_000_000


class SkillTimeout(Exception):
    pass


def bounded_range(*args):
    r = range(*args)
    if len(r) > MAX_RANGE:
        raise ValueError(f"range of {len(r)} exceeds the skill limit of {MAX_RANGE}")
    return r


def send(message):
    sys.stdout.write(json.dumps(message, default=str) + "\n")
    sys.stdout.flush()


def receive():
    line = sys.stdin.readline()
    if not line:
        raise SystemExit(0)
    return json.loads(line)


class Api:
    """`api.<tool>(...)` asks the parent to run the read-only tool and waits for its result."""

    def __init__(self, names):
        self.offered = frozenset(names)

    def __getattr__(self, name):
        if name not in self.__dict__.get("offered", ()):
            raise AttributeError(f"api has no {name!r}")

        def call(*args, **kwargs):
            send({"call": name, "args": list(args), "kwargs": kwargs})
            reply = receive()
            if "error" in reply:
                raise RuntimeError(reply["error"])
            return reply["result"]

        return call


def main():
    request = receive()
    name = request["name"]
    allowed = {n: getattr(builtins, n) for n in request["builtins"] if n != "range" and hasattr(builtins, n)}
    allowed["range"] = bounded_range
    namespace = {"__builtins__": allowed}
    deadline = time.monotonic() + float(request["timeout"])

    def tracer(frame, event, arg):
        if time.monotonic() > deadline:
            raise SkillTimeout(f"{name} took longer than {float(request['timeout']):g} seconds")
        return tracer

    try:
        exec(compile(request["source"], f"<skill:{name}>", "exec"), namespace)
        function = namespace[name]
        sys.settrace(tracer)
        try:
            value = function(api=Api(request["api"]), **request["kwargs"])
        finally:
            sys.settrace(None)
        send({"done": value})
    except SkillTimeout as exc:
        send({"failed": str(exc)})
    except Exception as exc:  # noqa: BLE001 - the skill's own bug, reported to the agent
        send({"failed": f"{name} failed: {type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()
