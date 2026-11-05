"""metacog serve: an agent's metacognition over HTTP, with a live dashboard.

    metacog serve app:agent [--context app:context_for] [--port 8620]

One process serves one deployment (one store) and is a serving process like any
other: runs it takes are recorded, judged and watched by the detector. Learning
happens wherever the deployment's `learning` setting puts it (with "worker",
in `metacog worker`). The dashboard at / shows runs as they land, what needs a
person, and each procedure's state, and acts through the API below.

    GET  /api/summary                         runs, verdicts and money (the economics report)
    GET  /api/runs?limit=50&before=N          recent runs, newest first
    GET  /api/runs/{run_id}                   one run: request, tool steps, answer, verdict, routing scores
    POST /api/runs                            run the agent: {"input", "case_id"?, "case_data"?, "context"?}
    POST /api/runs/{run_id}/feedback          a person's verdict: {"accepted", "reason"?}
    GET  /api/queue                           what needs a person: proposals, and runs awaiting a verdict
    GET  /api/procedures                      each procedure: versions, trust state, recent verdicts
    GET  /api/procedures/{name}/{version}     one version: source, provenance, gate result, diff from parent
    POST /api/procedures/{name}/{version}/approve
    POST /api/procedures/{name}/quarantine    {"reason"?}
    GET  /api/stream                          server-sent events: a message whenever the store changes

A run's context comes from `context(body)`, a factory the deployment provides
(`--context module:attribute`), or else is the body's "context" value as sent.

Every POST must carry the header `X-Metacog: 1`, which the dashboard sends. A
page on another site can't add that header without a CORS preflight, which this
service never grants, so it can't judge runs or approve code through a viewer's
browser. There is no other authentication: keep the default localhost binding,
or put the service behind your own.
"""

from __future__ import annotations

import asyncio
import difflib
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from importlib.resources import files
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from metacog.exceptions import AgentsException, UserError
from metacog.meta.report import summarize
from metacog.model_settings import DEFAULT_MODEL
from metacog.recording import MissingReply
from metacog.run import RunConfig, Runner


class RunRequest(BaseModel):
    input: str | list[dict[str, Any]]
    case_id: str | None = None
    case_data: Any = None
    context: Any = None


class FeedbackRequest(BaseModel):
    accepted: bool
    reason: str = ""


class QuarantineRequest(BaseModel):
    reason: str = "by hand, from the dashboard"


def _guard(x_metacog: str | None = Header(default=None)) -> None:
    if x_metacog != "1":
        raise HTTPException(403, "a POST needs the header X-Metacog: 1 (the dashboard sends it)")


def run_view(run: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    """A run as the API returns it; `full` adds the tool steps, the answer and the routing scores."""
    view = {key: run.get(key) for key in ("ord", "run_id", "case_id", "request", "model", "effort", "matched",
                                          "offered", "verdict", "verdict_source", "reason", "dollars",
                                          "reasoning_tokens", "seconds", "created_at")}
    view["called"] = run.get("called") or []
    if full:
        content = run.get("content") or {}
        view.update(steps=content.get("steps") or [], answer=content.get("answer"), scores=run.get("scores") or {})
    return view


def version_view(version: dict[str, Any]) -> dict[str, Any]:
    info = version.get("info") or {}
    gate = info.get("gate") or {}
    return {"name": version["name"], "version": version["version"], "state": version["state"],
            "created_by": version["created_by"], "parent": version["parent"], "created_at": version["created_at"],
            "learned_from": len(info.get("learned_from") or []), "writer_calls": info.get("writer_calls"),
            "dollars": info.get("dollars"), "gate": gate or None}


def create_app(agent: Any, *, context: Callable[[dict[str, Any]], Any] | None = None,
               run_config: RunConfig | None = None, max_turns: int = 12, poll: float = 0.5) -> FastAPI:
    """The service for one agent's deployment."""
    meta = getattr(agent, "metacognition", None)
    if meta is None or not hasattr(meta, "store"):
        raise UserError("metacog serve needs an agent with Metacognition (one store; PerUser isn't supported yet)")
    store = meta.store
    page = (files("metacog") / "dashboard.html").read_text(encoding="utf-8")
    app = FastAPI(title=f"metacog: {agent.name}", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")

    def found(run_id: str) -> dict[str, Any]:
        run = store.run(run_id)
        if run is None:
            raise HTTPException(404, f"no recorded run {run_id}")
        return run

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> str:
        return page

    @app.get("/api/summary")
    async def summary() -> dict[str, Any]:
        # The report's own "learning" holds learning's totals; the deployment's setting is learning_mode.
        return {**jsonable_encoder(summarize(store)), "agent": agent.name, "model": agent.model or DEFAULT_MODEL,
                "learning_mode": meta.learning, "approval": meta.approval}

    @app.get("/api/runs")
    async def runs(limit: int = 50, before: int | None = None) -> list[dict[str, Any]]:
        return [run_view(run) for run in store.recent_runs(min(max(limit, 1), 500), before=before)]

    @app.get("/api/runs/{run_id}")
    async def run(run_id: str) -> dict[str, Any]:
        return jsonable_encoder(run_view(found(run_id), full=True))

    @app.post("/api/runs", dependencies=[Depends(_guard)])
    async def start(body: RunRequest) -> dict[str, Any]:
        world = context(body.model_dump()) if context is not None else body.context
        config = replace(run_config or RunConfig(), case_id=body.case_id, case_data=body.case_data)
        try:
            result = await Runner.run(agent, body.input, context=world, max_turns=max_turns, run_config=config)
        except AgentsException as exc:
            raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc
        except MissingReply as exc:  # serving from a replay-only recording that lacks this call
            raise HTTPException(409, exc.args[0]) from exc
        report = result.metacognition
        return jsonable_encoder({"run_id": report.run_id, "final_output": result.final_output,
                                 "metacognition": {"effort": report.effort, "matched": report.matched,
                                                   "offered": report.offered, "called": report.called,
                                                   "verdict": report.verdict, "events": report.events}})

    @app.post("/api/runs/{run_id}/feedback", dependencies=[Depends(_guard)])
    async def feedback(run_id: str, body: FeedbackRequest) -> dict[str, Any]:
        found(run_id)
        events = await meta.feedback(run_id, body.accepted, body.reason)
        return {**run_view(found(run_id)), "events": events}

    @app.get("/api/queue")
    async def queue() -> dict[str, Any]:
        proposals = [version_view(v) for v in store.versions() if v["state"] == "proposed"]
        return {"proposals": proposals, "unjudged": [run_view(r) for r in store.recent_runs(50, unjudged=True)]}

    @app.get("/api/procedures")
    async def procedures() -> list[dict[str, Any]]:
        out = []
        for name, proc in meta.procedures.items():
            state = store.state(name)
            trusted = store.trusted(name)
            out.append({"name": name, "description": proc.description,
                        "trusted": trusted["version"] if trusted else None,
                        "recent": state.get("recent", []), "quarantine_reason": state.get("reason"),
                        "next_attempt": state.get("next_attempt"),
                        "versions": [version_view(v) for v in store.versions(name)]})
        return out

    @app.get("/api/procedures/{name}/{version}")
    async def version(name: str, version: int) -> dict[str, Any]:
        chosen = store.version(name, version)
        if chosen is None:
            raise HTTPException(404, f"{name} has no version {version}")
        base = store.version(name, chosen["parent"]) if chosen["parent"] else None
        base = base or next((v for v in reversed(store.versions(name)) if v["version"] < version), None)
        diff = "" if base is None else "".join(difflib.unified_diff(
            base["source"].splitlines(keepends=True), chosen["source"].splitlines(keepends=True),
            f"{name}@{base['version']}", f"{name}@{version}"))
        return {**version_view(chosen), "source": chosen["source"], "info": chosen["info"],
                "compared_with": None if base is None else base["version"], "diff": diff}

    @app.post("/api/procedures/{name}/{version}/approve", dependencies=[Depends(_guard)])
    async def approve(name: str, version: int) -> dict[str, Any]:
        if name not in meta.procedures:
            raise HTTPException(404, f"no procedure named {name}")
        try:
            meta.approve(name, version)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return version_view(store.version(name, version))

    @app.post("/api/procedures/{name}/quarantine", dependencies=[Depends(_guard)])
    async def quarantine(name: str, body: QuarantineRequest) -> dict[str, Any]:
        if name not in meta.procedures:
            raise HTTPException(404, f"no procedure named {name}")
        if store.trusted(name) is None:
            raise HTTPException(409, f"{name} has no trusted version")
        meta.quarantine(name, body.reason)
        return {"name": name, "trusted": None, "reason": body.reason}

    @app.get("/api/stream")
    async def stream(request: Request) -> StreamingResponse:
        async def events() -> AsyncIterator[str]:
            yield "retry: 2000\n\n"
            last, quiet = None, 0.0
            while not await request.is_disconnected():
                stamp = store.stamp()
                if stamp != last:
                    last, quiet = stamp, 0.0
                    yield f"data: {json.dumps({'stamp': stamp})}\n\n"
                elif quiet >= 15.0:
                    quiet = 0.0
                    yield ": still here\n\n"  # keeps proxies from closing an idle stream
                await asyncio.sleep(poll)
                quiet += poll

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
