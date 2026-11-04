"""Archive discovery endpoints: list, add by path, drop a manual entry."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from helix_web.server.models import ArchiveInfo, ArchiveList

router = APIRouter(tags=["archives"])


class AddArchiveBody(BaseModel):
    path: str


@router.get("/archives", response_model=ArchiveList)
async def list_archives(request: Request, rescan: bool = False) -> ArchiveList:
    registry = request.app.state.registry
    if rescan:
        registry.discover()
    return ArchiveList(archives=registry.list())


@router.post("/archives", response_model=ArchiveInfo, status_code=201)
async def add_archive(request: Request, body: AddArchiveBody) -> ArchiveInfo:
    registry = request.app.state.registry
    try:
        entry = registry.add(body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ArchiveInfo(**registry.describe(entry))


@router.delete("/archives/{key}", status_code=204)
async def remove_archive(request: Request, key: str) -> None:
    registry = request.app.state.registry
    try:
        registry.remove(key)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown archive: {key}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
