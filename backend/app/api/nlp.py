"""NLP / biomedical extraction API routes."""

from typing import Any

from fastapi import APIRouter, HTTPException, Path as PathParam, status

from app.database import get_connection
from app.logging import logger
from app.nlp.nlp_service import NLPService
from app.schemas import (
    EntityMentionOut,
    EntityOut,
    EvidenceOut,
    ProcessAllResponse,
    ProcessPaperResponse,
    ProcessingStatusOut,
    RelationshipOut,
    ReprocessResponse,
)

router = APIRouter(prefix="/investigations", tags=["nlp"])

_service: NLPService | None = None


def get_service() -> NLPService:
    global _service
    if _service is None:
        _service = NLPService()
    return _service


# ------------------------------------------------------------------
# Process a single paper
# ------------------------------------------------------------------
@router.post(
    "/{investigation_id}/nlp/process/{paper_id}",
    response_model=ProcessPaperResponse,
)
async def process_paper(
    investigation_id: int = PathParam(..., ge=1),
    paper_id: int = PathParam(..., ge=1),
) -> Any:
    """Process a single paper: extract entities, relationships, evidence."""
    # Verify investigation exists
    with get_connection() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

    # Verify paper exists
    service = get_service()
    paper = service.get_paper(investigation_id, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    result = service.process_paper(investigation_id, paper_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Processing failed"))
    return ProcessPaperResponse(**result)


# ------------------------------------------------------------------
# Process all papers in an investigation
# ------------------------------------------------------------------
@router.post(
    "/{investigation_id}/nlp/process-all",
    response_model=ProcessAllResponse,
)
async def process_all_papers(
    investigation_id: int = PathParam(..., ge=1),
) -> Any:
    """Process all papers in an investigation that have full text."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

    service = get_service()
    result = service.process_all_papers(investigation_id)
    return ProcessAllResponse(**result)


# ------------------------------------------------------------------
# Get entities
# ------------------------------------------------------------------
@router.get(
    "/{investigation_id}/nlp/entities",
    response_model=list[EntityOut],
)
def get_entities(
    investigation_id: int = PathParam(..., ge=1),
    entity_type: str | None = None,
) -> Any:
    """Get extracted entities for an investigation, optionally filtered by type."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

    service = get_service()
    entities = service.get_entities(investigation_id, entity_type)
    return [EntityOut(**e) for e in entities]


# ------------------------------------------------------------------
# Get entity mentions
# ------------------------------------------------------------------
@router.get(
    "/{investigation_id}/nlp/entities/{entity_id}/mentions",
    response_model=list[EntityMentionOut],
)
def get_entity_mentions(
    investigation_id: int = PathParam(..., ge=1),
    entity_id: int = PathParam(..., ge=1),
) -> Any:
    """Get all mentions of an entity across papers in an investigation."""
    with get_connection() as conn:
        # Verify entity belongs to investigation
        row = conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE entity_id = ? AND investigation_id = ?
            """,
            (entity_id, investigation_id),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        rows = conn.execute(
            """
            SELECT mention_id, entity_id, paper_id, section_id, chunk_id,
                   entity_name_original, page_number
            FROM paper_entity_mentions
            WHERE entity_id = ? AND investigation_id = ?
            ORDER BY paper_id, page_number
            """,
            (entity_id, investigation_id),
        ).fetchall()
        return [EntityMentionOut(**dict(r)) for r in rows]


# ------------------------------------------------------------------
# Get relationships
# ------------------------------------------------------------------
@router.get(
    "/{investigation_id}/nlp/relationships",
    response_model=list[RelationshipOut],
)
def get_relationships(
    investigation_id: int = PathParam(..., ge=1),
    relationship_type: str | None = None,
) -> Any:
    """Get extracted relationships for an investigation, optionally filtered by type."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

    service = get_service()
    relationships = service.get_relationships(investigation_id, relationship_type)
    return [RelationshipOut(**r) for r in relationships]


# ------------------------------------------------------------------
# Get evidence for a relationship
# ------------------------------------------------------------------
@router.get(
    "/{investigation_id}/nlp/relationships/{relationship_id}/evidence",
    response_model=list[EvidenceOut],
)
def get_relationship_evidence(
    investigation_id: int = PathParam(..., ge=1),
    relationship_id: int = PathParam(..., ge=1),
) -> Any:
    """Get all evidence passages supporting a relationship."""
    service = get_service()
    evidence = service.get_evidence(investigation_id, relationship_id)
    return [EvidenceOut(**e) for e in evidence]


# ------------------------------------------------------------------
# Get processing status
# ------------------------------------------------------------------
@router.get(
    "/{investigation_id}/nlp/status/{paper_id}",
    response_model=ProcessingStatusOut,
)
def get_processing_status(
    investigation_id: int = PathParam(..., ge=1),
    paper_id: int = PathParam(..., ge=1),
) -> Any:
    """Get NLP processing status for a paper."""
    service = get_service()
    status = service.get_processing_status(investigation_id, paper_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return ProcessingStatusOut(**status)


# ------------------------------------------------------------------
# Reprocess a paper
# ------------------------------------------------------------------
@router.post(
    "/{investigation_id}/nlp/reprocess/{paper_id}",
    response_model=ReprocessResponse,
)
async def reprocess_paper(
    investigation_id: int = PathParam(..., ge=1),
    paper_id: int = PathParam(..., ge=1),
) -> Any:
    """Reprocess a paper (clears existing NLP data first)."""
    service = get_service()
    paper = service.get_paper(investigation_id, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    result = service.reprocess_paper(investigation_id, paper_id)
    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Reprocessing failed"))
    return ReprocessResponse(**result)
