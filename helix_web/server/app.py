"""FastAPI application factory for the Helix Observatory.

The app is assembled here: routers, gzip, static files, and the exception
handler that turns sqlite lock contention into a retryable 503. Tests build
their own app against a temp Settings; run.py builds the real one.
"""

from __future__ import annotations

import mimetypes
import sqlite3

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from helix_web.server.config import Settings
from helix_web.server.db import is_locked_error
from helix_web.server.registry import ArchiveRegistry
from helix_web.server.routers import (
    archives,
    champions,
    diff,
    events,
    health,
    lineage,
    memory,
    rounds,
    summary,
    trajectories,
    variants,
    verdicts,
    writes,
)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    # Windows resolves MIME types from the registry, which may hand ES modules
    # a bogus type; declare the ones the app serves explicitly.
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("image/svg+xml", ".svg")

    app = FastAPI(title="Helix Observatory", version="0.1.0")
    app.state.settings = settings
    registry = ArchiveRegistry(settings)
    registry.discover()
    app.state.registry = registry

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.exception_handler(sqlite3.OperationalError)
    async def _locked_handler(request: Request, exc: sqlite3.OperationalError):
        if is_locked_error(exc):
            return JSONResponse(
                status_code=503,
                content={"detail": "archive is busy; retry shortly", "retry_after": 1},
                headers={"Retry-After": "1"},
            )
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    app.include_router(health.router, prefix="/api")
    app.include_router(archives.router, prefix="/api")
    app.include_router(summary.router, prefix="/api")
    app.include_router(variants.router, prefix="/api")
    app.include_router(lineage.router, prefix="/api")
    app.include_router(memory.router, prefix="/api")
    app.include_router(champions.router, prefix="/api")
    app.include_router(diff.router, prefix="/api")
    app.include_router(verdicts.router, prefix="/api")
    app.include_router(trajectories.router, prefix="/api")
    app.include_router(rounds.router, prefix="/api")
    app.include_router(writes.router, prefix="/api")
    app.include_router(events.router, prefix="/api")

    if settings.web_dir.exists():
        app.mount(
            "/",
            NoCacheStaticFiles(directory=str(settings.web_dir), html=True),
            name="web",
        )
    return app


class NoCacheStaticFiles(StaticFiles):
    """Static files that must revalidate on every request.

    Without an explicit Cache-Control, browsers apply heuristic freshness to
    ES modules and can serve a stale dashboard for hours after an edit, even
    across reloads. no-cache still allows 304s, so revalidation stays cheap.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response
