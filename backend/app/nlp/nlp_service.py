"""NLP service: orchestrates document processing, NER, relationship extraction, evidence storage."""

import hashlib
from datetime import datetime, timezone
from typing import Any

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    MAX_CHUNKS_PER_PAPER,
)
from app.database import get_connection, get_transaction
from app.logging import logger
from app.nlp.biomedical_ner import BiomedicalNER, ExtractedEntity
from app.nlp.document_processing import (
    extract_pdf_text,
    extract_txt_text,
    extract_xml_text,
    compute_passage_hash,
    clean_text,
    ExtractedChunk,
    ExtractedSection,
)
from app.nlp.relationship_extraction import (
    ExtractedRelationship,
    RelationshipExtractor,
)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NLPService:
    """Service for biomedical NLP pipeline: document processing, NER, relationship extraction."""

    def __init__(self):
        self.ner = BiomedicalNER(use_scispacy=False)  # Rule-based, SciSpaCy optional
        self.relationship_extractor = RelationshipExtractor(self.ner)
        logger.info("NLP service initialized")

    # ------------------------------------------------------------------
    # Process a single paper
    # ------------------------------------------------------------------
    def process_paper(self, investigation_id: int, paper_id: int) -> dict[str, Any]:
        """Process a single paper: extract sections, chunks, entities, relationships, evidence.

        Idempotent: clears existing NLP data for the paper before reprocessing.
        """
        from app.services.literature_service import LiteratureService

        lit_service = LiteratureService()
        paper = lit_service.get_paper(investigation_id, paper_id)
        if not paper:
            return {"success": False, "error": "Paper not found"}

        if paper["processing_status"] == "completed":
            return {
                "success": True,
                "message": "Already processed",
                "paper_id": paper_id,
                "processing_status": "completed",
            }

        full_text_path = paper.get("full_text_path")
        if not full_text_path:
            return {
                "success": False,
                "error": "No full text available for processing",
            }

        # Resolve full path
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent.parent.parent
        full_path = base_dir / full_text_path
        if not full_path.exists():
            return {
                "success": False,
                "error": f"Full text file not found: {full_text_path}",
            }

        # Mark as processing
        self._update_processing_status(investigation_id, paper_id, "processing")

        try:
            # Extract text, sections, chunks based on file type
            ext = full_path.suffix.lower()
            if ext == ".pdf":
                full_text, sections, chunks = extract_pdf_text(str(full_path))
            elif ext == ".txt":
                full_text, sections, chunks = extract_txt_text(str(full_path))
            elif ext == ".xml":
                full_text, sections, chunks = extract_xml_text(str(full_path))
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {ext}",
                }

            full_text = clean_text(full_text)

            # Store sections and chunks in database
            self._store_sections_and_chunks(
                investigation_id, paper_id, sections, chunks
            )

            # Extract entities from chunks
            all_entities: list[ExtractedEntity] = []
            for chunk in chunks:
                chunk_entities = self.ner.extract_entities(
                    chunk.chunk_text,
                    page_number=chunk.page_start,
                    section_name=None,
                    chunk_id=None,  # Will be set after chunk storage
                )
                # Update chunk_id in entities after we store chunks
                for ce in chunk_entities:
                    ce.chunk_id = None  # Placeholder
                all_entities.extend(chunk_entities)

            # Store entities and mentions
            stored_entities = self._store_entities(
                investigation_id, paper_id, sections, chunks, all_entities
            )

            # Extract relationships
            all_relationships: list[ExtractedRelationship] = []
            for chunk in chunks:
                chunk_entities = [
                    e for e in stored_entities
                    if e.entity_name_original.lower() in chunk.chunk_text.lower()
                ]
                if len(chunk_entities) < 2:
                    continue

                chunk_relationships = self.relationship_extractor.extract_relationships(
                    chunk.chunk_text,
                    chunk_entities,
                    page_number=chunk.page_start,
                    section_name=None,
                    chunk_id=None,
                )
                all_relationships.extend(chunk_relationships)

            # Store relationships and evidence
            stored_relationships = self._store_relationships_and_evidence(
                investigation_id, paper_id, all_relationships
            )

            # Mark as completed
            self._update_processing_status(investigation_id, paper_id, "completed")

            logger.info(
                "Processed paper %d: %d entities, %d relationships",
                paper_id, len(stored_entities), len(stored_relationships),
            )

            return {
                "success": True,
                "paper_id": paper_id,
                "processing_status": "completed",
                "entities_extracted": len(stored_entities),
                "relationships_extracted": len(stored_relationships),
                "sections_found": len(sections),
                "chunks_created": len(chunks),
            }

        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to process paper %d: %s", paper_id, exc)
            self._update_processing_status(
                investigation_id, paper_id, "failed", str(exc)
            )
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Process all papers in an investigation
    # ------------------------------------------------------------------
    def process_all_papers(self, investigation_id: int) -> dict[str, Any]:
        """Process all papers in an investigation that have full text."""
        from app.services.literature_service import LiteratureService

        lit_service = LiteratureService()
        papers = lit_service.list_papers(investigation_id)

        results = {
            "investigation_id": investigation_id,
            "total_papers": len(papers),
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "paper_results": [],
        }

        for paper in papers:
            if paper["access_status"] not in ("downloaded", "user_imported", "open_access_available"):
                results["skipped"] += 1
                results["paper_results"].append({
                    "paper_id": paper["paper_id"],
                    "status": "skipped",
                    "reason": f"access_status={paper['access_status']}",
                })
                continue

            if paper["processing_status"] == "completed":
                results["skipped"] += 1
                results["paper_results"].append({
                    "paper_id": paper["paper_id"],
                    "status": "skipped",
                    "reason": "already_processed",
                })
                continue

            result = self.process_paper(investigation_id, paper["paper_id"])
            if result["success"]:
                results["processed"] += 1
            else:
                results["failed"] += 1
            results["paper_results"].append({
                "paper_id": paper["paper_id"],
                "status": result.get("processing_status", "failed"),
                "entities": result.get("entities_extracted", 0),
                "relationships": result.get("relationships_extracted", 0),
                "error": result.get("error"),
            })

        return results

    # ------------------------------------------------------------------
    # Get entities for an investigation
    # ------------------------------------------------------------------
    def get_entities(self, investigation_id: int, entity_type: str | None = None) -> list[dict[str, Any]]:
        """Get all entities for an investigation, optionally filtered by type."""
        with get_connection() as conn:
            if entity_type:
                rows = conn.execute(
                    """
                    SELECT e.entity_id, e.normalized_name, e.entity_type,
                           e.external_identifier, e.external_identifier_source,
                           COUNT(m.mention_id) as mention_count
                    FROM entities e
                    LEFT JOIN paper_entity_mentions m ON e.entity_id = m.entity_id
                      AND e.investigation_id = m.investigation_id
                    WHERE e.investigation_id = ? AND e.entity_type = ?
                    GROUP BY e.entity_id
                    ORDER BY mention_count DESC, e.normalized_name
                    """,
                    (investigation_id, entity_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT e.entity_id, e.normalized_name, e.entity_type,
                           e.external_identifier, e.external_identifier_source,
                           COUNT(m.mention_id) as mention_count
                    FROM entities e
                    LEFT JOIN paper_entity_mentions m ON e.entity_id = m.entity_id
                      AND e.investigation_id = m.investigation_id
                    WHERE e.investigation_id = ?
                    GROUP BY e.entity_id
                    ORDER BY mention_count DESC, e.normalized_name
                    """,
                    (investigation_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Get relationships for an investigation
    # ------------------------------------------------------------------
    def get_relationships(
        self,
        investigation_id: int,
        relationship_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get all relationships for an investigation, optionally filtered by type."""
        with get_connection() as conn:
            if relationship_type:
                rows = conn.execute(
                    """
                    SELECT r.relationship_id, r.source_entity_id, r.target_entity_id,
                           r.relationship_type, r.confidence_score,
                           r.is_direct_interaction, r.negation, r.uncertainty,
                           r.speculation, r.experimental_context,
                           r.source_passage, r.source_passage_hash,
                           r.created_at,
                           se.normalized_name as source_name,
                           te.normalized_name as target_name
                    FROM relationships r
                    JOIN entities se ON r.source_entity_id = se.entity_id
                      AND r.investigation_id = se.investigation_id
                    JOIN entities te ON r.target_entity_id = te.entity_id
                      AND r.investigation_id = te.investigation_id
                    WHERE r.investigation_id = ? AND r.relationship_type = ?
                    ORDER BY r.confidence_score DESC
                    """,
                    (investigation_id, relationship_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT r.relationship_id, r.source_entity_id, r.target_entity_id,
                           r.relationship_type, r.confidence_score,
                           r.is_direct_interaction, r.negation, r.uncertainty,
                           r.speculation, r.experimental_context,
                           r.source_passage, r.source_passage_hash,
                           r.extraction_method, r.created_at,
                           se.normalized_name as source_name,
                           te.normalized_name as target_name
                    FROM relationships r
                    JOIN entities se ON r.source_entity_id = se.entity_id
                      AND r.investigation_id = se.investigation_id
                    JOIN entities te ON r.target_entity_id = te.entity_id
                      AND r.investigation_id = te.investigation_id
                    WHERE r.investigation_id = ?
                    ORDER BY r.confidence_score DESC
                    """,
                    (investigation_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Get evidence for a relationship
    # ------------------------------------------------------------------
    def get_evidence(
        self,
        investigation_id: int,
        relationship_id: int,
    ) -> list[dict[str, Any]]:
        """Get all evidence for a relationship."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT e.evidence_id, e.relationship_id, e.paper_id, e.section_id,
                       e.supporting_text, e.page_number, e.evidence_type,
                       e.extraction_method, e.confidence_score, e.extracted_at,
                       r.source_passage, r.source_passage_hash
                FROM evidence e
                JOIN relationships r ON e.relationship_id = r.relationship_id
                  AND e.investigation_id = r.investigation_id
                WHERE e.investigation_id = ? AND e.relationship_id = ?
                ORDER BY e.extracted_at
                """,
                (investigation_id, relationship_id),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Get processing status
    # ------------------------------------------------------------------
    def get_processing_status(self, investigation_id: int, paper_id: int) -> dict[str, Any] | None:
        """Get processing status for a paper."""
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT paper_id, processing_status, download_error
                FROM papers
                WHERE paper_id = ? AND investigation_id = ?
                """,
                (paper_id, investigation_id),
            ).fetchone()
            return dict(row) if row else None

    def get_paper(self, investigation_id: int, paper_id: int) -> dict[str, Any] | None:
        """Get a single paper by ID (delegates to LiteratureService)."""
        from app.services.literature_service import LiteratureService
        lit_service = LiteratureService()
        return lit_service.get_paper(investigation_id, paper_id)

    # ------------------------------------------------------------------
    # Reprocess a paper
    # ------------------------------------------------------------------
    def reprocess_paper(self, investigation_id: int, paper_id: int) -> dict[str, Any]:
        """Reprocess a paper (clears existing NLP data first)."""
        self._clear_paper_nlp_data(investigation_id, paper_id)
        return self.process_paper(investigation_id, paper_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _update_processing_status(
        self,
        investigation_id: int,
        paper_id: int,
        status: str,
        error: str | None = None,
    ) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE papers
                SET processing_status = ?, download_error = ?
                WHERE paper_id = ? AND investigation_id = ?
                """,
                (status, error, paper_id, investigation_id),
            )
            conn.commit()

    def _store_sections_and_chunks(
        self,
        investigation_id: int,
        paper_id: int,
        sections: list,
        chunks: list,
    ) -> None:
        """Store sections and chunks in database."""
        with get_transaction() as conn:
            # Clear existing sections/chunks for this paper
            conn.execute(
                "DELETE FROM paper_chunks WHERE paper_id = ? AND investigation_id = ?",
                (paper_id, investigation_id),
            )
            conn.execute(
                "DELETE FROM paper_sections WHERE paper_id = ? AND investigation_id = ?",
                (paper_id, investigation_id),
            )

            # Store sections
            section_id_map: dict[int, int] = {}  # index -> section_id
            for idx, section in enumerate(sections):
                cur = conn.execute(
                    """
                    INSERT INTO paper_sections
                    (paper_id, investigation_id, section_name, section_order, page_start, page_end)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id, investigation_id,
                        section.section_name, section.section_order,
                        section.page_start, section.page_end,
                    ),
                )
                section_id_map[idx] = cur.lastrowid

            # Store chunks
            for chunk in chunks:
                section_id = section_id_map.get(
                    chunk.section_id if chunk.section_id is not None else 0
                ) if chunk.section_id is not None else None
                # Find matching section by page range
                if section_id is None and sections:
                    # Assign to first section that contains the page
                    for s_idx, s in enumerate(sections):
                        if s.page_start <= chunk.page_start <= s.page_end:
                            section_id = section_id_map.get(s_idx)
                            break

                conn.execute(
                    """
                    INSERT INTO paper_chunks
                    (paper_id, investigation_id, section_id, chunk_text,
                     chunk_order, page_start, page_end, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id, investigation_id, section_id,
                        chunk.chunk_text, chunk.chunk_order,
                        chunk.page_start, chunk.page_end, chunk.token_count,
                    ),
                )
            conn.commit()

    def _store_entities(
        self,
        investigation_id: int,
        paper_id: int,
        sections: list,
        chunks: list,
        entities: list,
    ) -> list:
        """Store entities and their mentions in database. Returns stored entity objects."""
        with get_transaction() as conn:
            # Clear existing entities/mentions for this paper
            conn.execute(
                """
                DELETE FROM paper_entity_mentions
                WHERE paper_id = ? AND investigation_id = ?
                """,
                (paper_id, investigation_id),
            )
            # Note: entities are shared across papers in an investigation,
            # so we don't delete them globally - just the mentions.

            stored_entities = []
            for ent in entities:
                # Check if entity already exists (by normalized_name + type + investigation)
                existing = conn.execute(
                    """
                    SELECT entity_id FROM entities
                    WHERE investigation_id = ? AND normalized_name = ? AND entity_type = ?
                    """,
                    (investigation_id, ent.normalized_name, ent.entity_type),
                ).fetchone()

                if existing:
                    entity_id = existing["entity_id"]
                    # Update synonyms if original name differs
                    if ent.entity_name_original != ent.normalized_name:
                        self._add_synonym(conn, entity_id, investigation_id, ent.entity_name_original)
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO entities
                        (investigation_id, normalized_name, entity_type,
                         external_identifier, external_identifier_source)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            investigation_id, ent.normalized_name, ent.entity_type,
                            ent.external_identifier, ent.external_identifier_source,
                        ),
                    )
                    entity_id = cur.lastrowid
                    # Add original name as synonym
                    self._add_synonym(conn, entity_id, investigation_id, ent.entity_name_original)

                # Store mention
                conn.execute(
                    """
                    INSERT INTO paper_entity_mentions
                    (paper_id, investigation_id, section_id, entity_id,
                     chunk_id, entity_name_original, page_number)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id, investigation_id, None,  # section_id - we don't track precisely
                        entity_id, ent.chunk_id, ent.entity_name_original,
                        ent.page_number,
                    ),
                )

                stored_entities.append(type("Entity", (), {
                    "entity_id": entity_id,
                    "entity_name_original": ent.entity_name_original,
                    "normalized_name": ent.normalized_name,
                    "entity_type": ent.entity_type,
                    "confidence": ent.confidence,
                })())

            conn.commit()
            return stored_entities

    def _add_synonym(
        self,
        conn,
        entity_id: int,
        investigation_id: int,
        synonym: str,
    ) -> None:
        """Add a synonym for an entity if not already present."""
        existing = conn.execute(
            """
            SELECT synonym_id FROM entity_synonyms
            WHERE entity_id = ? AND investigation_id = ? AND synonym = ?
            """,
            (entity_id, investigation_id, synonym),
        ).fetchone()
        if not existing:
            conn.execute(
                """
                INSERT INTO entity_synonyms
                (entity_id, investigation_id, synonym, source)
                VALUES (?, ?, ?, ?)
                """,
                (entity_id, investigation_id, synonym, "extraction"),
            )

    def _store_relationships_and_evidence(
        self,
        investigation_id: int,
        paper_id: int,
        relationships: list,
    ) -> list:
        """Store relationships and their evidence in database."""
        with get_transaction() as conn:
            stored = []
            for rel in relationships:
                # Get source and target entity IDs
                source_entity_id = self._get_entity_id_by_name(
                    conn, investigation_id, rel.source_entity.normalized_name,
                    rel.source_entity.entity_type,
                )
                target_entity_id = self._get_entity_id_by_name(
                    conn, investigation_id, rel.target_entity.normalized_name,
                    rel.target_entity.entity_type,
                )

                if source_entity_id is None or target_entity_id is None:
                    continue

                # Check if relationship already exists (idempotent)
                existing = conn.execute(
                    """
                    SELECT relationship_id FROM relationships
                    WHERE investigation_id = ?
                      AND source_entity_id = ?
                      AND target_entity_id = ?
                      AND relationship_type = ?
                      AND source_passage_hash = ?
                    """,
                    (
                        investigation_id, source_entity_id, target_entity_id,
                        rel.relationship_type, rel.source_passage_hash,
                    ),
                ).fetchone()

                if existing:
                    # Update evidence only
                    rel_id = existing["relationship_id"]
                else:
                    cur = conn.execute(
                        """
                        INSERT INTO relationships
                        (investigation_id, source_entity_id, target_entity_id,
                         relationship_type, confidence_score, extraction_method,
                         is_direct_interaction, negation, uncertainty, speculation,
                         experimental_context, source_passage, source_passage_hash,
                         created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            investigation_id, source_entity_id, target_entity_id,
                            rel.relationship_type, rel.confidence, "rule",
                            1 if rel.is_direct_interaction else 0,
                            1 if rel.negation else 0,
                            1 if rel.uncertainty else 0,
                            1 if rel.speculation else 0,
                            rel.experimental_context,
                            rel.source_passage, rel.source_passage_hash,
                            _iso_now(),
                        ),
                    )
                    rel_id = cur.lastrowid

                # Store evidence
                conn.execute(
                    """
                    INSERT INTO evidence
                    (relationship_id, investigation_id, paper_id, section_id,
                     supporting_text, page_number, evidence_type,
                     extraction_method, confidence_score, extracted_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        rel_id, investigation_id, paper_id, None,
                        rel.source_passage, rel.page_number,
                        "experimental" if rel.experimental_context else "other",
                        "rule", rel.confidence, _iso_now(),
                    ),
                )

                stored.append({
                    "relationship_id": rel_id,
                    "source_entity": rel.source_entity.normalized_name,
                    "target_entity": rel.target_entity.normalized_name,
                    "relationship_type": rel.relationship_type,
                    "confidence": rel.confidence,
                })

            conn.commit()
            return stored

    def _get_entity_id_by_name(
        self,
        conn,
        investigation_id: int,
        name: str,
        entity_type: str,
    ) -> int | None:
        """Get entity ID by normalized name and type."""
        row = conn.execute(
            """
            SELECT entity_id FROM entities
            WHERE investigation_id = ? AND normalized_name = ? AND entity_type = ?
            """,
            (investigation_id, name, entity_type),
        ).fetchone()
        return row["entity_id"] if row else None

    def _clear_paper_nlp_data(
        self,
        investigation_id: int,
        paper_id: int,
    ) -> None:
        """Clear all NLP data for a paper (for reprocessing)."""
        with get_transaction() as conn:
            # Get entity IDs mentioned in this paper
            entity_rows = conn.execute(
                """
                SELECT DISTINCT entity_id FROM paper_entity_mentions
                WHERE paper_id = ? AND investigation_id = ?
                """,
                (paper_id, investigation_id),
            ).fetchall()
            entity_ids = [r["entity_id"] for r in entity_rows]

            # Delete evidence for relationships involving these entities
            for eid in entity_ids:
                conn.execute(
                    """
                    DELETE FROM evidence
                    WHERE investigation_id = ?
                      AND relationship_id IN (
                        SELECT r.relationship_id FROM relationships r
                        WHERE r.investigation_id = ?
                          AND (r.source_entity_id = ? OR r.target_entity_id = ?)
                      )
                    """,
                    (investigation_id, investigation_id, eid, eid),
                )

            # Delete relationships involving these entities
            for eid in entity_ids:
                conn.execute(
                    """
                    DELETE FROM relationships
                    WHERE investigation_id = ?
                      AND (source_entity_id = ? OR target_entity_id = ?)
                    """,
                    (investigation_id, eid, eid),
                )

            # Delete mentions
            conn.execute(
                """
                DELETE FROM paper_entity_mentions
                WHERE paper_id = ? AND investigation_id = ?
                """,
                (paper_id, investigation_id),
            )

            # Delete chunks and sections
            conn.execute(
                "DELETE FROM paper_chunks WHERE paper_id = ? AND investigation_id = ?",
                (paper_id, investigation_id),
            )
            conn.execute(
                "DELETE FROM paper_sections WHERE paper_id = ? AND investigation_id = ?",
                (paper_id, investigation_id),
            )

            conn.commit()
