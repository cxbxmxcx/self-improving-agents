"""metacog: inspect and steer a deployment's metacognition store, and learn for it.

    metacog status STORE                       procedures, trust states, and economics
    metacog runs STORE [--last N]              recent runs: model, effort, procedures offered and called,
                                               verdict, cost
    metacog show STORE NAME [--version N]      one procedure version's source and provenance
    metacog approve STORE NAME VERSION         trust a version the gate proposed
    metacog quarantine STORE NAME [--reason]   stop trusting a procedure; it relearns from new runs
    metacog report STORE [--html FILE]         the economics, as text or a standalone HTML page
    metacog worker TARGET [--once]             learn for an agent whose metacognition has learning="worker"

STORE is the directory given to Metacognition(store=...), or a TARGET. A TARGET names the agent,
as module:attribute (the Agent, or a function that returns one), imported from the current directory.
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
import re
import sys
from pathlib import Path

from metacog.meta.report import render, summarize
from metacog.meta.runtime import approve_version, quarantine_procedure
from metacog.meta.store import MetaStore

TARGET = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_]\w*$")


def _open(where: str) -> MetaStore:
    if TARGET.match(where) and not Path(where).exists():
        from metacog.meta.worker import load_agent

        store = getattr(load_agent(where).metacognition, "store", None)
        if store is None:
            raise SystemExit(f"{where} has no metacognition store")
        return store
    if not (Path(where) / "meta.sqlite").is_file():
        raise SystemExit(f"no metacognition store at {where} (expected {Path(where) / 'meta.sqlite'})")
    return MetaStore(where)


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
    from dotenv import load_dotenv

    from metacog.meta.worker import load_agent, work

    load_dotenv(Path.cwd() / ".env")  # the worker's writer calls are paid calls
    agent = load_agent(args.store)
    try:
        asyncio.run(work(agent, once=args.once, interval=args.interval, lease=args.lease))
    except KeyboardInterrupt:
        print("worker stopped")


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
    parser = argparse.ArgumentParser(prog="metacog", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fn, extra in (("status", _status, None), ("runs", _runs, "runs"), ("show", _show, "show"),
                            ("approve", _approve, "approve"), ("quarantine", _quarantine, "quarantine"),
                            ("report", _report, "report"), ("worker", _worker, "worker")):
        command = sub.add_parser(name)
        command.add_argument("store", metavar="TARGET" if name == "worker" else "STORE")
        command.set_defaults(fn=fn)
        if extra == "worker":
            command.add_argument("--once", action="store_true", help="one pass, then exit")
            command.add_argument("--interval", type=float, default=30.0, help="seconds between passes")
            command.add_argument("--lease", type=float, default=300.0, help="seconds the worker's lease lasts")
        elif extra == "runs":
            command.add_argument("--last", type=int, default=20)
        elif extra in ("show", "approve", "quarantine"):
            command.add_argument("name")
            if extra == "show":
                command.add_argument("--version", type=int)
            elif extra == "approve":
                command.add_argument("version", type=int)
            else:
                command.add_argument("--reason", default="by hand")
                command.add_argument("--relearn-after", type=int, default=4)
        elif extra == "report":
            command.add_argument("--html")
    args = parser.parse_args(argv)
    try:
        args.fn(args)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
