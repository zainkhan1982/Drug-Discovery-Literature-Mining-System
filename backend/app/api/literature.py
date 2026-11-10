"""Literature acquisition API routes."""

import os
from typing import Any

from fastapi import APIRouter, File, HTTPException, Path as PathParam, UploadFile, status

from app.database import get_connection
from app.logging import logger
from app.schemas import (
    DownloadResponse,
    ImportResponse,
    LiteratureSearchRequest,
    LiteratureSearchResponse,
    PaperListResponse,
    PaperOut,
)
from app.services.literature_service import LiteratureService

router = APIRouter(prefix="/investigations", tags=["literature"])

_service: LiteratureService | None = None


def get_service() -> LiteratureService:
    global _service
    if _service is None:
        _service = LiteratureService()
    return _service


@router.on_event("shutdown")
async def _shutdown() -> None:
    global _service
    if _service is not None:
        await _service.close()
        _service = None


# ------------------------------------------------------------------
# Search
# ------------------------------------------------------------------
@router.post(
    "/{investigation_id}/literature/search",
    response_model=LiteratureSearchResponse,
)
async def search_literature(
    investigation_id: int = PathParam(..., ge=1),
    request: LiteratureSearchRequest = None,
) -> Any:
    """Search literature sources and store new papers for an investigation."""
    # Verify investigation exists
    with get_connection() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

    req = request or LiteratureSearchRequest()
    sources = req.sources or ["pubmed", "europepmc"]

    # Validate sources
    valid_sources = {"pubmed", "europepmc"}
    invalid = set(sources) - valid_sources
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid sources: {', '.join(invalid)}. Valid: {', '.join(valid_sources)}",
        )

    date_range = None
    if req.start_date or req.end_date:
        date_range = {"start": req.start_date or "", "end": req.end_date or ""}

    try:
        service = get_service()
        result = await service.search_literature(
            investigation_id=investigation_id,
            query=req.query,
            sources=sources,
            max_papers=req.max_papers or 100,
            date_range=date_range,
            publication_types=req.publication_types,
            open_access_only=req.open_access_only,
        )
        return LiteratureSearchResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Literature search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ------------------------------------------------------------------
# List papers
# ------------------------------------------------------------------
@router.get(
    "/{investigation_id}/literature",
    response_model=PaperListResponse,
)
def list_literature(
    investigation_id: int = PathParam(..., ge=1),
) -> Any:
    """List all papers for an investigation."""
    # Verify investigation exists
    with get_connection() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

    service = get_service()
    papers = service.list_papers(investigation_id)
    count = service.get_paper_count(investigation_id)
    limit = 100  # from config

    return PaperListResponse(
        papers=[PaperOut(**p) for p in papers],
        count=count,
        limit=limit,
        remaining=max(0, limit - count),
    )


# ------------------------------------------------------------------
# Get single paper
# ------------------------------------------------------------------
@router.get(
    "/{investigation_id}/literature/{paper_id}",
    response_model=PaperOut,
)
def get_literature_paper(
    investigation_id: int = PathParam(..., ge=1),
    paper_id: int = PathParam(..., ge=1),
) -> Any:
    """Get a single paper by ID."""
    service = get_service()
    paper = service.get_paper(investigation_id, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return PaperOut(**paper)


# ------------------------------------------------------------------
# Download full text
# ------------------------------------------------------------------
@router.post(
    "/{investigation_id}/literature/{paper_id}/download",
    response_model=DownloadResponse,
)
async def download_paper_full_text(
    investigation_id: int = PathParam(..., ge=1),
    paper_id: int = PathParam(..., ge=1),
) -> Any:
    """Download full text for a paper if open access is available."""
    service = get_service()
    paper = service.get_paper(investigation_id, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    result = await service.download_full_text(investigation_id, paper_id)
    return DownloadResponse(**result)


# ------------------------------------------------------------------
# Import file (PDF/TXT)
# ------------------------------------------------------------------
@router.post(
    "/{investigation_id}/literature/import",
    response_model=ImportResponse,
)
async def import_literature_file(
    investigation_id: int = PathParam(..., ge=1),
    file: UploadFile = File(...),
) -> Any:
    """Import a user-uploaded PDF or TXT file as a paper."""
    # Verify investigation exists
    with get_connection() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

    # Read file content
    content = await file.read()

    service = get_service()
    result = service.import_file(
        investigation_id=investigation_id,
        file_bytes=content,
        original_filename=file.filename or "uploaded_file",
    )

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Import failed"))

    return ImportResponse(**result)


# ------------------------------------------------------------------
# Delete paper
# ------------------------------------------------------------------
@router.delete(
    "/{investigation_id}/literature/{paper_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_literature_paper(
    investigation_id: int = PathParam(..., ge=1),
    paper_id: int = PathParam(..., ge=1),
) -> None:
    """Delete a paper from an investigation."""
    service = get_service()
    success = service.delete_paper(investigation_id, paper_id)
    if not success:
        raise HTTPException(status_code=404, detail="Paper not found")
