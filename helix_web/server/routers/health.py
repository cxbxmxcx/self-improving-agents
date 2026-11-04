"""Liveness endpoint, also handy as a smoke target in tests."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(request: Request) -> dict:
    registry = request.app.state.registry
    settings = request.app.state.settings
    return {
        "ok": True,
        "version": request.app.version,
        "n_archives": len(registry.keys()),
        "poll_ms": settings.poll_ms,
    }
