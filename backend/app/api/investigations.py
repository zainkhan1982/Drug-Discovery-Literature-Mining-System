"""Investigation CRUD endpoints."""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Path as PathParam

from app.database import get_connection, get_transaction, init_database
from app.logging import logger
from app.schemas import (
    InvestigationCreate,
    InvestigationOut,
    InvestigationUpdate,
)

router = APIRouter(prefix="/investigations", tags=["investigations"])


@router.post("", response_model=InvestigationOut, status_code=201)
def create_investigation(payload: InvestigationCreate) -> Any:
    """Create a new research investigation."""
    import json
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    search_params: Dict[str, Any] = {
        "disease": payload.disease,
        "drug": payload.drug,
        "gene": payload.gene,
        "start_date": payload.start_date,
        "end_date": payload.end_date,
        "publication_types": payload.publication_types,
        "full_text_preference": payload.full_text_preference,
        "max_papers": payload.max_papers,
    }

    with get_transaction() as conn:
        cur = conn.execute(
            """
            INSERT INTO investigations (research_question, search_parameters, status, created_at, updated_at)
            VALUES (?, ?, 'queued', ?, ?)
            """,
            (payload.research_question, json.dumps(search_params), now, now),
        )
        investigation_id = cur.lastrowid

        row = conn.execute(
            "SELECT * FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()

    logger.info("Created investigation %d: %s", investigation_id, payload.research_question[:80])
    return _row_to_out(row)


@router.get("", response_model=List[InvestigationOut])
def list_investigations() -> Any:
    """List all investigations."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM investigations ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_out(row) for row in rows]


@router.get("/{investigation_id}", response_model=InvestigationOut)
def get_investigation(investigation_id: int = PathParam(..., ge=1)) -> Any:
    """Get a single investigation by ID."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return _row_to_out(row)


@router.patch("/{investigation_id}", response_model=InvestigationOut)
def update_investigation(
    investigation_id: int = PathParam(..., ge=1),
    payload: InvestigationUpdate = None,
) -> Any:
    """Update an investigation (e.g., status)."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    with get_transaction() as conn:
        row = conn.execute(
            "SELECT * FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")

        update_fields = {}
        if payload and payload.status is not None:
            update_fields["status"] = payload.status
        if update_fields:
            update_fields["updated_at"] = now
            set_clause = ", ".join(f"{k} = ?" for k in update_fields)
            values = list(update_fields.values()) + [investigation_id]
            conn.execute(
                f"UPDATE investigations SET {set_clause} WHERE investigation_id = ?",
                values,
            )

        row = conn.execute(
            "SELECT * FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
    return _row_to_out(row)


@router.delete("/{investigation_id}", status_code=204)
def delete_investigation(investigation_id: int = PathParam(..., ge=1)) -> None:
    """Delete an investigation and all its related data (cascade via FKs)."""
    with get_transaction() as conn:
        row = conn.execute(
            "SELECT investigation_id FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Investigation not found")
        conn.execute(
            "DELETE FROM investigations WHERE investigation_id = ?",
            (investigation_id,),
        )


def _row_to_out(row: Any) -> InvestigationOut:
    import json

    return InvestigationOut(
        investigation_id=row["investigation_id"],
        research_question=row["research_question"],
        search_parameters=json.loads(row["search_parameters"]) if row["search_parameters"] else {},
        status=row["status"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
