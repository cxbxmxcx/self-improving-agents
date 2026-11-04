"""The two write endpoints: promote a version, purge failed trajectories."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from helix_web.server import writes
from helix_web.server.deps import resolve_archive

router = APIRouter(tags=["writes"])


class PromoteBody(BaseModel):
    artifact_id: str
    version: int
    approver: str = "dashboard"
    reason: str = "manual promotion via dashboard"


class PurgeBody(BaseModel):
    confirm: bool = False


@router.post("/archives/{key}/promote")
async def promote(request: Request, key: str, body: PromoteBody) -> dict:
    entry = resolve_archive(request, key)
    try:
        record = await writes.promote(
            entry.path, body.artifact_id, body.version, body.approver, body.reason
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return record


@router.post("/archives/{key}/trajectories/purge")
async def purge(request: Request, key: str, body: PurgeBody) -> dict:
    if not body.confirm:
        raise HTTPException(status_code=400, detail="pass confirm=true to purge")
    entry = resolve_archive(request, key)
    deleted = await writes.purge_failed(entry.path)
    return {"deleted": deleted}
