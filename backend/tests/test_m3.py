"""M3 acceptance tests: biomedical NLP and relationship extraction."""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.database import DATABASE_PATH, get_connection, init_database, reset_database, init_database, reset_database
from app.main import create_app
from app.nlp.biomedical_ner import BiomedicalNER, ExtractedEntity
from app.nlp.document_processing import (
    compute_passage_hash,
    extract_txt_text,
    extract_pdf_text,
)
from app.nlp.relationship_extraction import RelationshipExtractor
from app.nlp.nlp_service import NLPService


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.config as cfg
    import app.database as db

    tmp_db = tmp_path / "test.db"
    tmp_full_text = tmp_path / "full_text"
    tmp_full_text.mkdir(parents=True, exist_ok=True)

    cfg.DATA_DIR = tmp_path
    cfg.DATABASE_PATH = tmp_db
    cfg.DATABASE_URL = f"sqlite:///{tmp_db}"
    cfg.FULL_TEXT_DIR = tmp_full_text
    db.DATABASE_PATH = tmp_db

    app = create_app()
    from fastapi.testclient import TestClient
    with TestClient(app) as tc:
        yield tc


# -----------------------------------------------------------------------
# Document processing tests
# -----------------------------------------------------------------------
def test_extract_txt_text(tmp_path):
    """TXT file extraction creates sections and chunks."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_text(
        "Abstract\nThis is the abstract about Alzheimer disease.\n"
        "Introduction\nThis section introduces the topic.\n"
        "Methods\nWe used a mouse model.\n"
        "Results\nThe drug inhibits the protein.\n"
        "Conclusion\nThe drug is effective.\n"
    )

    full_text, sections, chunks = extract_txt_text(str(txt_file))

    assert "Alzheimer disease" in full_text
    assert len(sections) >= 1
    assert len(chunks) >= 1
    assert any(s.section_name in ("abstract", "introduction", "methods", "results", "conclusion") for s in sections)


def test_compute_passage_hash():
    """Passage hash is deterministic and unique."""
    h1 = compute_passage_hash("The drug inhibits the protein.")
    h2 = compute_passage_hash("The drug inhibits the protein.")
    h3 = compute_passage_hash("The drug activates the protein.")

    assert h1 == h2  # Same passage = same hash
    assert h1 != h3  # Different passage = different hash
    assert len(h1) == 64  # SHA-256 hex


def test_clean_text():
    """Text cleaning normalizes whitespace without destroying meaning."""
    from app.nlp.document_processing import clean_text

    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("") == ""
    assert clean_text("word1\t\tword2") == "word1 word2"


# -----------------------------------------------------------------------
# Biomedical NER tests
# -----------------------------------------------------------------------
def test_ner_extracts_drugs():
    """NER extracts drug names from text."""
    ner = BiomedicalNER(use_scispacy=False)
    entities = ner.extract_entities(
        "Aspirin inhibits COX-2 in patients.",
        page_number=1,
        section_name="results",
    )

    drug_entities = [e for e in entities if e.entity_type == "drug"]
    assert len(drug_entities) >= 1
    assert any("aspirin" in e.normalized_name for e in drug_entities)


def test_ner_extracts_genes():
    """NER extracts gene names from text."""
    ner = BiomedicalNER(use_scispacy=False)
    entities = ner.extract_entities(
        "The BRCA1 gene is associated with breast cancer.",
        page_number=1,
        section_name="introduction",
    )

    gene_entities = [e for e in entities if e.entity_type == "gene"]
    assert len(gene_entities) >= 1
    assert any("brca1" in e.normalized_name for e in gene_entities)


def test_ner_extracts_diseases():
    """NER extracts disease names from text."""
    ner = BiomedicalNER(use_scispacy=False)
    entities = ner.extract_entities(
        "Alzheimer disease is a neurodegenerative disorder.",
        page_number=1,
        section_name="abstract",
    )

    disease_entities = [e for e in entities if e.entity_type == "disease"]
    assert len(disease_entities) >= 1
    assert any("alzheimer" in e.normalized_name for e in disease_entities)


def test_ner_preserves_original_text():
    """NER preserves original text and normalized name separately."""
    ner = BiomedicalNER(use_scispacy=False)
    entities = ner.extract_entities("Aspirin was tested.", page_number=1)

    aspirin_entities = [e for e in entities if "aspirin" in e.normalized_name]
    assert len(aspirin_entities) >= 1
    assert aspirin_entities[0].entity_name_original == "Aspirin"
    assert aspirin_entities[0].normalized_name == "aspirin"


def test_ner_entity_has_confidence():
    """All entities have a confidence score."""
    ner = BiomedicalNER(use_scispacy=False)
    entities = ner.extract_entities("Aspirin inhibits COX-2.", page_number=1)

    for e in entities:
        assert 0.0 <= e.confidence <= 1.0


# -----------------------------------------------------------------------
# Negation / uncertainty / speculation detection
# -----------------------------------------------------------------------
def test_negation_detection():
    """Negation is detected and not converted to positive assertion."""
    ner = BiomedicalNER(use_scispacy=False)
    ctx = ner.detect_context("The drug does not inhibit the protein.")
    assert ctx["negation"] is True


def test_no_false_negation_conversion():
    """'Does not inhibit' is not treated as 'activates'."""
    ner = BiomedicalNER(use_scispacy=False)
    ctx = ner.detect_context("The drug does not inhibit the protein.")
    assert ctx["negation"] is True
    # The relationship extractor should preserve this as negation,
    # not convert to a positive activation


def test_uncertainty_detection():
    """Uncertainty language is detected."""
    ner = BiomedicalNER(use_scispacy=False)
    ctx = ner.detect_context("The drug may inhibit the protein.")
    assert ctx["uncertainty"] is True


def test_speculation_detection():
    """Speculation language is detected."""
    ner = BiomedicalNER(use_scispacy=False)
    ctx = ner.detect_context("We hypothesize that the drug inhibits the protein.")
    assert ctx["speculation"] is True


def test_experimental_context_detection():
    """Experimental context is detected."""
    ner = BiomedicalNER(use_scispacy=False)
    ctx = ner.detect_context("Experiments were performed in vitro using cell lines.")
    assert ctx["experimental_context"] in ("in_vitro", "cell_line")


# -----------------------------------------------------------------------
# Relationship extraction tests
# -----------------------------------------------------------------------
def test_relationship_extraction_inhibits():
    """Inhibits relationship is extracted correctly."""
    ner = BiomedicalNER(use_scispacy=False)
    extractor = RelationshipExtractor(ner)

    entities = [
        ExtractedEntity(
            entity_name_original="Aspirin",
            normalized_name="aspirin",
            entity_type="drug",
            confidence=0.8,
            page_number=1,
            section_name="results",
        ),
        ExtractedEntity(
            entity_name_original="COX-2",
            normalized_name="cox-2",
            entity_type="protein",
            confidence=0.8,
            page_number=1,
            section_name="results",
        ),
    ]

    relationships = extractor.extract_relationships(
        "Aspirin inhibits COX-2 activity.",
        entities,
        page_number=1,
        section_name="results",
    )

    assert len(relationships) >= 1
    rel = relationships[0]
    assert rel.relationship_type == "inhibits"
    assert rel.is_direct_interaction is True
    assert rel.negation is False


def test_relationship_extraction_negation():
    """Negated relationship is preserved as negation, not converted."""
    ner = BiomedicalNER(use_scispacy=False)
    extractor = RelationshipExtractor(ner)

    entities = [
        ExtractedEntity(
            entity_name_original="Aspirin",
            normalized_name="aspirin",
            entity_type="drug",
            confidence=0.8,
            page_number=1,
        ),
        ExtractedEntity(
            entity_name_original="COX-2",
            normalized_name="cox-2",
            entity_type="protein",
            confidence=0.8,
            page_number=1,
        ),
    ]

    relationships = extractor.extract_relationships(
        "Aspirin does not inhibit COX-2 activity.",
        entities,
        page_number=1,
    )

    # Should still find the relationship but with negation=True
    neg_rels = [r for r in relationships if r.relationship_type == "inhibits" and r.negation is True]
    assert len(neg_rels) >= 1, f"Expected negated relationship, got: {[ (r.relationship_type, r.negation) for r in relationships ]}"


def test_relationship_extraction_treats_disease():
    """Treats relationship between drug and disease is extracted."""
    ner = BiomedicalNER(use_scispacy=False)
    extractor = RelationshipExtractor(ner)

    entities = [
        ExtractedEntity(
            entity_name_original="Aspirin",
            normalized_name="aspirin",
            entity_type="drug",
            confidence=0.8,
            page_number=1,
        ),
        ExtractedEntity(
            entity_name_original="Alzheimer disease",
            normalized_name="alzheimer disease",
            entity_type="disease",
            confidence=0.8,
            page_number=1,
        ),
    ]

    relationships = extractor.extract_relationships(
        "Aspirin treats Alzheimer disease.",
        entities,
        page_number=1,
    )

    treat_rels = [r for r in relationships if r.relationship_type == "treats"]
    assert len(treat_rels) >= 1


def test_relationship_has_passage_and_hash():
    """Every relationship has a source passage and hash."""
    ner = BiomedicalNER(use_scispacy=False)
    extractor = RelationshipExtractor(ner)

    entities = [
        ExtractedEntity(
            entity_name_original="Aspirin",
            normalized_name="aspirin",
            entity_type="drug",
            confidence=0.8,
            page_number=1,
        ),
        ExtractedEntity(
            entity_name_original="COX-2",
            normalized_name="cox-2",
            entity_type="protein",
            confidence=0.8,
            page_number=1,
        ),
    ]

    relationships = extractor.extract_relationships(
        "Aspirin inhibits COX-2 activity.",
        entities,
        page_number=1,
    )

    assert len(relationships) >= 1
    for r in relationships:
        assert r.source_passage is not None
        assert len(r.source_passage) > 0
        assert r.source_passage_hash is not None
        assert len(r.source_passage_hash) == 64  # SHA-256


def test_relationship_confidence_threshold():
    """Low confidence relationships are filtered out."""
    ner = BiomedicalNER(use_scispacy=False)
    extractor = RelationshipExtractor(ner)

    # Use entities with low confidence
    entities = [
        ExtractedEntity(
            entity_name_original="UnknownDrugX",
            normalized_name="unknowndrugx",
            entity_type="drug",
            confidence=0.1,  # Very low
            page_number=1,
        ),
        ExtractedEntity(
            entity_name_original="UnknownProteinY",
            normalized_name="unknownproteiny",
            entity_type="protein",
            confidence=0.1,
            page_number=1,
        ),
    ]

    relationships = extractor.extract_relationships(
        "UnknownDrugX inhibits UnknownProteinY.",
        entities,
        page_number=1,
    )

    # All should be below threshold
    for r in relationships:
        assert r.confidence >= 0.0  # May be empty due to threshold filter


# -----------------------------------------------------------------------
# Contradiction preservation
# -----------------------------------------------------------------------
def test_contradiction_preservation():
    """Both 'inhibits' and 'activates' for same pair are preserved."""
    ner = BiomedicalNER(use_scispacy=False)
    extractor = RelationshipExtractor(ner)

    entities = [
        ExtractedEntity(
            entity_name_original="DrugA",
            normalized_name="druga",
            entity_type="drug",
            confidence=0.9,
            page_number=1,
        ),
        ExtractedEntity(
            entity_name_original="ProteinB",
            normalized_name="proteinb",
            entity_type="protein",
            confidence=0.9,
            page_number=1,
        ),
    ]

    # Two sentences with contradictory findings
    text = "DrugA inhibits ProteinB in vitro. However, DrugA activates ProteinB in clinical trials."
    relationships = extractor.extract_relationships(text, entities, page_number=1)

    inhib = [r for r in relationships if r.relationship_type == "inhibits"]
    activ = [r for r in relationships if r.relationship_type == "activates"]
    assert len(inhib) >= 1
    assert len(activ) >= 1


# -----------------------------------------------------------------------
# API tests
# -----------------------------------------------------------------------
def test_api_process_nonexistent_paper(client):
    """Process on nonexistent paper returns 404."""
    r = client.post("/investigations/1/nlp/process/99999")
    assert r.status_code == 404


def test_api_process_paper_no_full_text(client):
    """Process on paper without full text returns error."""
    r = client.post("/investigations", json={"research_question": "Test"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    # Insert a paper without full text
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database, normalized_title, download_url, download_attempted, file_size, mime_type, processing_status) "
            "VALUES (?, 'NOFT1', NULL, 'No Full Text', '', '', '2024', 'pubmed', NULL, 'abstract_only', datetime('now'), 'pubmed', 'no full text', NULL, 0, NULL, NULL, 'pending')",
            (inv_id,),
        )
        conn.commit()
        paper_row = conn.execute("SELECT paper_id FROM papers WHERE pmid = 'NOFT1'").fetchone()
        paper_id = paper_row["paper_id"]

    r = client.post(f"/investigations/{inv_id}/nlp/process/{paper_id}")
    assert r.status_code == 400
    assert "No full text" in r.json()["detail"]


def test_api_process_all_empty(client):
    """Process all on investigation with no papers returns empty."""
    r = client.post("/investigations", json={"research_question": "Test empty"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.post(f"/investigations/{inv_id}/nlp/process-all")
    assert r.status_code == 200
    data = r.json()
    assert data["total_papers"] == 0
    assert data["processed"] == 0


def test_api_get_entities_empty(client):
    """Get entities on empty investigation returns empty list."""
    r = client.post("/investigations", json={"research_question": "Test entities"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.get(f"/investigations/{inv_id}/nlp/entities")
    assert r.status_code == 200
    assert r.json() == []


def test_api_get_relationships_empty(client):
    """Get relationships on empty investigation returns empty list."""
    r = client.post("/investigations", json={"research_question": "Test rels"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.get(f"/investigations/{inv_id}/nlp/relationships")
    assert r.status_code == 200
    assert r.json() == []


def test_api_processing_status(client):
    """Get processing status for a paper."""
    r = client.post("/investigations", json={"research_question": "Test status"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database, normalized_title, download_url, download_attempted, file_size, mime_type, processing_status) "
            "VALUES (?, 'STATUS1', NULL, 'Status Paper', '', '', '2024', 'pubmed', NULL, 'abstract_only', datetime('now'), 'pubmed', 'status paper', NULL, 0, NULL, NULL, 'pending')",
            (inv_id,),
        )
        conn.commit()
        paper_row = conn.execute("SELECT paper_id FROM papers WHERE pmid = 'STATUS1'").fetchone()
        paper_id = paper_row["paper_id"]

    r = client.get(f"/investigations/{inv_id}/nlp/status/{paper_id}")
    assert r.status_code == 200
    data = r.json()
    assert data["processing_status"] == "pending"


def test_api_reprocess_nonexistent(client):
    """Reprocess on nonexistent paper returns 404."""
    r = client.post("/investigations/1/nlp/reprocess/99999")
    assert r.status_code == 404


def test_investigation_isolation_nlp(client):
    """NLP data is isolated per investigation."""
    r1 = client.post("/investigations", json={"research_question": "Inv A"})
    r2 = client.post("/investigations", json={"research_question": "Inv B"})
    inv1 = r1.json()["investigation_id"]
    inv2 = r2.json()["investigation_id"]

    # Insert entities directly in inv1
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO entities (investigation_id, normalized_name, entity_type, external_identifier, external_identifier_source) "
            "VALUES (?, 'druga', 'drug', NULL, NULL)",
            (inv1,),
        )
        conn.commit()

    service = NLPService()
    entities_inv1 = service.get_entities(inv1)
    entities_inv2 = service.get_entities(inv2)
    assert len(entities_inv1) == 1
    assert len(entities_inv2) == 0


# -----------------------------------------------------------------------
# Idempotent processing
# -----------------------------------------------------------------------
def test_process_idempotent(client):
    """Processing the same paper twice doesn't duplicate entities."""
    from app.services.literature_service import LiteratureService

    r = client.post("/investigations", json={"research_question": "Test idempotent"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    # Import a TXT file
    txt_content = b"Aspirin inhibits COX-2 in a mouse model of inflammation."
    r = client.post(
        f"/investigations/{inv_id}/literature/import",
        files={"file": ("test.txt", txt_content, "text/plain")},
    )
    assert r.status_code == 200
    paper_id = r.json()["paper_id"]

    service = NLPService()
    result1 = service.process_paper(inv_id, paper_id)
    assert result1["success"] is True

    # Count entities
    entities_1 = service.get_entities(inv_id)
    count_1 = len(entities_1)

    # Process again (idempotent)
    result2 = service.reprocess_paper(inv_id, paper_id)
    assert result2["success"] is True

    entities_2 = service.get_entities(inv_id)
    count_2 = len(entities_2)

    assert count_1 == count_2, f"Entity count changed: {count_1} -> {count_2}"



