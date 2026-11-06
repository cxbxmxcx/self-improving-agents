"""metacog: build, inspect, steer and serve a deployment's metacognition.

    metacog status STORE                       procedures, trust states, and economics
    metacog runs STORE [--last N]              recent runs: model, effort, procedures offered and called,
                                               verdict, cost
    metacog show STORE NAME [--version N]      one procedure version's source and provenance
    metacog approve STORE NAME VERSION         trust a version the gate proposed
    metacog install STORE NAME FILE [--by]     trust a procedure written by hand (or found elsewhere)
    metacog quarantine STORE NAME [--reason]   stop trusting a procedure; it relearns from new runs
    metacog report STORE [--html FILE]         the economics, as text or a standalone HTML page
    metacog discover STORE [--draft]           recurring, costly work in the runs: candidates for a procedure
    metacog feedback TARGET RUN accept|reject  a person's verdict on a run (RUN: its id or its number)
    metacog calibrate TARGET CASES             measure the floors (and routing) on a case file; record them
    metacog serve TARGET                       the HTTP API and the live dashboard
    metacog worker TARGET [--once]             learn for an agent whose metacognition has learning="worker"

STORE is the directory given to Metacognition(store=...), or a TARGET. A TARGET names the agent,
as module:attribute (the Agent, or a function that returns one), imported from the current directory.
Commands that make model calls (calibrate, serve, worker, discover --draft) read keys from ./.env.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

from metacog.exceptions import UserError
from metacog.meta.report import render, summarize
from metacog.meta.runtime import approve_version, quarantine_procedure
from metacog.meta.store import MetaStore

TARGET = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


def _is_target(where: str) -> bool:
    return bool(TARGET.match(where)) and not Path(where).exists()


def _deployment(where: str) -> tuple[Any, MetaStore]:
    """The agent (None for a bare store path) and its store."""
    if _is_target(where):
        from metacog.meta.worker import load_agent

        agent = load_agent(where)
        store = getattr(agent.metacognition, "store", None)
        if store is None:
            raise SystemExit(f"{where} has no metacognition store")
        return agent, store
    if not (Path(where) / "meta.sqlite").is_file():
        raise SystemExit(f"no metacognition store at {where} (expected {Path(where) / 'meta.sqlite'})")
    return None, MetaStore(where)


def _open(where: str) -> MetaStore:
    return _deployment(where)[1]


def _env() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env")


def _status(args: argparse.Namespace) -> None:
    store = _open(args.store)
    print(render(summarize(store)))
    proposals = [v for v in store.versions() if v["state"] == "proposed"]
    for p in proposals:
        print(f"  awaiting approval: {p['name']}@{p['version']} (metacog approve {args.store} {p['name']} "
              f"{p['version']})")


def _procedures(run: dict) -> str:
    """The procedures a run was offered, with a * on each one it called."""
    called = set(run.get("called") or [])
    return ", ".join(f"{item}*" if item in called else item for item in run["offered"])


def _runs(args: argparse.Namespace) -> None:
    store = _open(args.store)
    runs = store.runs()[-args.last:]
    print(f"{'#':>4} {'case':<14} {'model':<18} {'effort':<7} {'offered (* called)':<30} {'verdict':<9} "
          f"{'dollars':>8}")
    for run in runs:
        verdict = run["verdict"] or "pending"
        print(f"{run['ord'] + 1:>4} {run['case_id'][:14]:<14} {(run.get('model') or '-')[:18]:<18}"
              f" {run['effort'] or '-':<7} {_procedures(run)[:30] or '-':<30} {verdict:<9} {run['dollars'] or 0:>8.4f}")


def _show(args: argparse.Namespace) -> None:
    store = _open(args.store)
    versions = store.versions(args.name)
    if not versions:
        raise SystemExit(f"no procedure named {args.name}")
    found = next((v for v in versions if v["version"] == args.version), None) if args.version else versions[-1]
    if found is None:
        raise SystemExit(f"{args.name} has no version {args.version}")
    print(f"{found['name']}@{found['version']}  {found['state']}  by {found['created_by']}"
          f"  parent {found['parent'] or '-'}  {found['created_at']}")
    if found["info"]:
        print(json.dumps(found["info"], indent=2))
    print("\n" + found["source"])


def _approve(args: argparse.Namespace) -> None:
    approve_version(_open(args.store), args.name, args.version)
    print(f"{args.name}@{args.version} is trusted")


def _mechanisms(args: argparse.Namespace) -> None:
    store = _open(args.store)
    designs = store.setting("mechanisms") or {}
    chosen = store.setting("learners") or {}
    if not designs and not chosen:
        print("no designed mechanisms and no learners chosen by a tournament")
    for name, design in designs.items():
        test = "-" if design.get("test") is None else f"{design['test']:.2f}"
        print(f"{name}  {design['state']}  from {design['parent']}  train {design['train']:.2f}  test {test}  "
              f"{design['created_at']}")
    for procedure, choice in chosen.items():
        print(f"{procedure} learns by {choice['mechanism']} ({choice['reason']}; {choice['chosen_at']})")


def _approve_mechanism(args: argparse.Namespace) -> None:
    from metacog.design import approve_mechanism

    approve_mechanism(_open(args.store), args.name)
    print(f"{args.name} may compete in learner tournaments")


def _install(args: argparse.Namespace) -> None:
    from metacog.meta.procedure import _check_source
    from metacog.meta.runtime import trust_version

    source = Path(args.source).read_text(encoding="utf-8")
    _check_source(source, args.name)  # the code-skill guard, and a def with the procedure's name
    store = _open(args.store)
    version = trust_version(store, args.name, store.add_version(args.name, source, state="candidate",
                                                                created_by=args.by))
    print(f"{args.name}@{version} is trusted (by {args.by}, from {args.source})")


def _quarantine(args: argparse.Namespace) -> None:
    if quarantine_procedure(_open(args.store), args.name, args.reason, relearn_after=args.relearn_after):
        print(f"{args.name} is quarantined; it relearns from runs recorded from now on")
    else:
        print(f"{args.name} has no trusted version")


def _report(args: argparse.Namespace) -> None:
    store = _open(args.store)
    summary = summarize(store)
    if not args.html:
        print(render(summary))
        return
    Path(args.html).write_text(html_report(store, summary), encoding="utf-8")
    print(f"wrote {args.html}")


def _worker(args: argparse.Namespace) -> None:
    from metacog.meta.events import webhook
    from metacog.meta.worker import load_agent, work

    _env()  # the worker's writer calls are paid calls
    agent = load_agent(args.store)
    if args.webhook:
        webhook(args.webhook)
    try:
        asyncio.run(work(agent, once=args.once, interval=args.interval, lease=args.lease))
    except KeyboardInterrupt:
        print("worker stopped")


def _feedback(args: argparse.Namespace) -> None:
    if not _is_target(args.store):
        raise SystemExit("feedback needs the agent (module:attribute), so its detector and learner can act on it")
    agent, store = _deployment(args.store)
    run = store.recent_runs(1, before=int(args.run)) if args.run.isdigit() else None
    run_id = run[0]["run_id"] if run and run[0]["ord"] == int(args.run) - 1 else args.run
    asyncio.run(agent.metacognition.feedback(run_id, args.verdict == "accept", args.reason))
    judged = store.run(run_id)
    print(f"run {judged['ord'] + 1} ({run_id}) is {judged['verdict']}"
          + (f": {judged['reason']}" if judged["reason"] else ""))


def _calibrate(args: argparse.Namespace) -> None:
    from metacog.meta.worker import load_agent, load_object
    from metacog.offline import calibrate_deployment, load_cases
    from metacog.recording import Recording

    _env()
    agent = load_agent(args.store)
    cases = load_cases(args.cases)
    world = load_object(args.world) if args.world else None
    recording = (Recording(args.recording, mode=args.mode, distinct_repeats=args.repeats > 1)
                 if args.recording else None)

    async def measure() -> Any:
        options = dict(world=world, efforts=args.efforts, arms=args.arms, repeats=args.repeats,
                       concurrency=args.concurrency, max_turns=args.max_turns)
        if recording is None:
            return await calibrate_deployment(agent, cases, **options)
        with recording.active():
            return await calibrate_deployment(agent, cases, **options)

    result = asyncio.run(measure())
    floors = result.floors
    print(f"{len(cases)} cases on {floors.model}")
    print(f"  {'arm':<10} {'effort':<7} {'solved':>7} {'reasoning':>9} {'dollars':>8}")
    for row in floors.rows():
        print(f"  {row['arm']:<10} {row['effort']:<7} {row['solved']:>3}/{row['total']:<3} "
              f"{row['reasoning_tokens']:>9.0f} {row['dollars']:>8.4f}")
    print(f"recorded floors {floors.table()} for {floors.model}")
    for note in result.skipped:
        print(f"skipped {note}")
    if result.routing is not None and result.routing.router.thresholds:
        thresholds = {name: p["threshold"] for name, p in result.routing.procedures.items()}
        print(f"recorded routing: {result.routing.accuracy:.0%} of labeled cases routed right; "
              f"thresholds {thresholds}")
    if recording is not None and recording.live_calls:
        from metacog.pricing import PRICES

        print(f"{recording.live_calls} live calls, ${recording.spent(PRICES):.4f}, saved to {recording.path}")


def _discover(args: argparse.Namespace) -> None:
    from metacog.discover import discover, draft_stub, skeleton
    from metacog.model_settings import DEFAULT_MODEL

    agent, store = _deployment(args.store)
    procedures = list(agent.metacognition.procedures) if agent is not None else []
    read_only = [t.name for t in agent.tools if getattr(t, "read_only", False)] if agent is not None else None
    groups = discover(store, procedures=procedures, min_runs=args.min_runs)[: args.top]
    if not groups:
        print(f"no work recurs in {args.min_runs} or more runs without a procedure")
        return
    if args.draft:
        _env()
    for n, group in enumerate(groups, 1):
        print(f"{n}. {', '.join(group.tools)}: {group.runs} runs, {group.accepted} of {group.judged} judged "
              f"accepted, {group.mean_reasoning_tokens:.0f} reasoning tokens a run, ${group.dollars:.4f} in all")
        for request in group.requests[:3]:
            print(f"     e.g. {request}")
        stub = (asyncio.run(draft_stub(group, store, model=args.model or getattr(agent, "model", None)
                                       or DEFAULT_MODEL, read_only=read_only))
                if args.draft else skeleton(group, read_only))
        print(textwrap.indent(stub, "     "))
    if read_only is None:
        print("(from a store path, every tool is listed in api=[...]; name the agent to keep only read-only ones)")


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from metacog.meta.events import webhook
    from metacog.meta.worker import load_agent, load_object
    from metacog.recording import Recording
    from metacog.run import RunConfig
    from metacog.service import create_app

    _env()
    agent = load_agent(args.store)
    context = load_object(args.context) if args.context else None
    if args.webhook:
        webhook(args.webhook)
    config = RunConfig(recording=Recording(args.recording, mode=args.mode)) if args.recording else None
    print(f"metacog: serving {agent.name} at http://{args.host}:{args.port}"
          + (f", model calls from {args.recording} ({args.mode})" if args.recording else ""))
    uvicorn.run(create_app(agent, context=context, run_config=config, max_turns=args.max_turns), host=args.host,
                port=args.port, log_level="warning")


def html_report(store: MetaStore, summary: dict) -> str:
    """A standalone page: economics, procedure versions, learning attempts, and the run timeline."""
    esc = html.escape

    def table(head: list[str], rows: list[list[str]]) -> str:
        return ("<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>"
                + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows) + "</table>")

    effort_rows = [[esc(e), f"{c['accepted']}/{c['judged']}", str(c["runs"]), f"{c['mean_reasoning_tokens']:.0f}",
                    f"${c['mean_dollars']:.4f}"] for e, c in summary["by_effort"].items()]
    version_rows = [[esc(v["name"]), str(v["version"]), f"<span class='s {esc(v['state'])}'>{esc(v['state'])}</span>",
                     esc(v["created_by"]), str(v["parent"] or "-"), esc(v["created_at"][:19])]
                    for v in store.versions()]
    learning_rows = [[esc(a["name"]), esc(a["mechanism"]), str(a["episodes"]), str(a["writer_calls"]),
                      f"${a['dollars']:.4f}", esc(a["outcome"]), str(a["at_ord"])] for a in store.learning()]
    run_rows = [[str(r["ord"] + 1), esc(r["case_id"]), esc(r.get("model") or "-"), esc(r["effort"] or "-"),
                 esc(_procedures(r) or "-"),
                 f"<span class='s {esc(r['verdict'] or 'pending')}'>{esc(r['verdict'] or 'pending')}</span>",
                 f"${(r['dollars'] or 0):.4f}"] for r in store.runs()]
    learning = summary["learning"]
    break_even = summary["break_even_requests"]
    pays_back = f" Break-even after about {break_even:.1f} requests with a procedure." if break_even else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Metacognition report</title><style>
:root{{--bg:#fff;--fg:#1d1d1f;--mute:#6e6e73;--line:#e5e5ea;--ok:#1a7f37;--bad:#c62828;--warn:#9a6700}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#f2f2f7;--mute:#a1a1a6;--line:#2c2c2e;
--ok:#4cc26a;--bad:#ff6b6b;--warn:#e3b341}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:960px;
padding:24px 16px}}h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px}}p{{color:var(--mute)}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;display:block;overflow-x:auto}}
th,td{{border-bottom:1px solid var(--line);padding:6px 10px;text-align:left;white-space:nowrap}}
.s{{font-weight:600}}.trusted,.accepted{{color:var(--ok)}}.quarantined,.rejected,.refused{{color:var(--bad)}}
.proposed,.pending{{color:var(--warn)}}</style></head><body>
<h1>Metacognition report</h1>
<p>{summary['runs']} runs. Learning: {learning['attempts']} attempts, {learning['writer_calls']} writer calls,
${learning['dollars']:.4f}.{pays_back}</p>
<h2>By reasoning setting</h2>{table(["effort", "accepted", "runs", "reasoning tokens", "dollars a run"], effort_rows)}
<h2>Procedure versions</h2>{table(["procedure", "version", "state", "by", "parent", "created"], version_rows)}
<h2>Learning attempts</h2>{table(["procedure", "mechanism", "runs", "writer calls", "dollars", "outcome", "at run"],
                                 learning_rows)}
<h2>Runs</h2>{table(["#", "case", "model", "effort", "offered (* called)", "verdict", "dollars"], run_rows)}
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    from metacog.meta.parts import EFFORTS

    parser = argparse.ArgumentParser(prog="metacog", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name: str, fn: Any, where: str = "STORE") -> argparse.ArgumentParser:
        found = sub.add_parser(name)
        found.add_argument("store", metavar=where)
        found.set_defaults(fn=fn)
        return found

    command("status", _status)
    command("runs", _runs).add_argument("--last", type=int, default=20)
    show = command("show", _show)
    show.add_argument("name")
    show.add_argument("--version", type=int)
    approve = command("approve", _approve)
    approve.add_argument("name")
    approve.add_argument("version", type=int)
    command("mechanisms", _mechanisms)
    command("approve-mechanism", _approve_mechanism).add_argument("name")
    install = command("install", _install)
    install.add_argument("name")
    install.add_argument("source", help="a .py file holding the procedure's function, taking `api` first")
    install.add_argument("--by", default="human", help="who wrote it: 'human' runs in-process, others sandboxed")
    quarantine = command("quarantine", _quarantine)
    quarantine.add_argument("name")
    quarantine.add_argument("--reason", default="by hand")
    quarantine.add_argument("--relearn-after", type=int, default=4)
    command("report", _report).add_argument("--html")
    found = command("discover", _discover)
    found.add_argument("--top", type=int, default=3, help="how many groups of runs to show")
    found.add_argument("--min-runs", type=int, default=2, help="the fewest runs a group needs")
    found.add_argument("--draft", action="store_true", help="have a model draft each stub (one call per group)")
    found.add_argument("--model", help="the model that drafts (default: the agent's)")
    feedback = command("feedback", _feedback, "TARGET")
    feedback.add_argument("run", help="a run id, or the run's number as `metacog runs` shows it")
    feedback.add_argument("verdict", choices=["accept", "reject"])
    feedback.add_argument("--reason", default="")
    calibrate = command("calibrate", _calibrate, "TARGET")
    calibrate.add_argument("cases", help="a JSONL case file: {id, input, data?, procedure?} a line")
    calibrate.add_argument("--world", help="module:attribute that builds a case's context from its data")
    calibrate.add_argument("--efforts", nargs="+", default=list(EFFORTS))
    calibrate.add_argument("--arms", nargs="+", default=["no_memory", "memory"], choices=["no_memory", "memory"])
    calibrate.add_argument("--repeats", type=int, default=1)
    calibrate.add_argument("--concurrency", type=int, default=4)
    calibrate.add_argument("--max-turns", type=int, default=10)
    calibrate.add_argument("--recording", help="serve replies from this JSONL recording, adding new ones")
    calibrate.add_argument("--mode", choices=["replay", "record", "live"], default="record")
    serve = command("serve", _serve, "TARGET")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8620)
    serve.add_argument("--context", help="module:attribute that builds a run's context from the request body")
    serve.add_argument("--webhook", help="post quarantines, proposals and installs to this URL")
    serve.add_argument("--max-turns", type=int, default=12)
    serve.add_argument("--recording", help="serve model replies from this JSONL recording (see --mode)")
    serve.add_argument("--mode", choices=["replay", "record", "live"], default="record")
    worker = command("worker", _worker, "TARGET")
    worker.add_argument("--once", action="store_true", help="one pass, then exit")
    worker.add_argument("--interval", type=float, default=30.0, help="seconds between passes")
    worker.add_argument("--lease", type=float, default=300.0, help="seconds the worker's lease lasts")
    worker.add_argument("--webhook", help="post proposals, installs and refusals to this URL")
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except KeyError as exc:
        print(exc.args[0] if exc.args else exc, file=sys.stderr)
        return 1
    except (ValueError, UserError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
