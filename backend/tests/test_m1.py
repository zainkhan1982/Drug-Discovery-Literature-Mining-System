"""M1 acceptance tests: repository structure, config, database, API."""

import json
import os
import sys
import tempfile

import pytest

# Ensure backend app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import DATA_DIR, DATABASE_PATH, MAX_PAPERS_PER_INVESTIGATION, OLLAMA_HOST
from app.database import DATABASE_PATH as DB_PATH2, get_connection, init_database, reset_database
from app.logging import logger
from app.main import create_app
from app.schemas import HealthOut


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with an isolated temp database."""
    # Point DATA_DIR and DATABASE_PATH to a temp location
    import app.config as cfg
    import app.database as db

    tmp_db = tmp_path / "test.db"
    cfg.DATA_DIR = tmp_path
    cfg.DATABASE_PATH = tmp_db
    cfg.DATABASE_URL = f"sqlite:///{tmp_db}"
    db.DATABASE_PATH = tmp_db

    app = create_app()
    from fastapi.testclient import TestClient
    with TestClient(app) as tc:
        yield tc


def test_config_values():
    """Configuration loads from env with correct defaults."""
    assert MAX_PAPERS_PER_INVESTIGATION == 100
    assert OLLAMA_HOST == "http://127.0.0.1:11434"
    assert str(DATABASE_PATH).endswith("data/app.db")


def test_database_init_and_fk(client):
    """Database initializes with foreign keys enabled and zero violations."""
    with get_connection() as conn:
        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert len(violations) == 0

        # schema_version table exists
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        assert row[0] is not None


def test_cross_investigation_isolation(client):
    """Same PMID allowed across investigations; duplicate within one rejected."""
    # Create two investigations
    r1 = client.post("/investigations", json={"research_question": "Alzheimer targets"})
    r2 = client.post("/investigations", json={"research_question": "Cancer targets"})
    assert r1.status_code == 201
    assert r2.status_code == 201
    inv1 = r1.json()["investigation_id"]
    inv2 = r2.json()["investigation_id"]

    with get_connection() as conn:
        # Insert same PMID in both investigations - should be allowed
        conn.execute(
            "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database) "
            "VALUES (?, '12345', '10.1/t1', 'Paper A', '[]', 'abs', '2020', 'pubmed', 'p1.pdf', 'open_access', datetime('now'), 'pubmed')",
            (inv1,),
        )
        conn.execute(
            "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database) "
            "VALUES (?, '12345', '10.1/t2', 'Paper B', '[]', 'abs', '2021', 'pubmed', 'p2.pdf', 'open_access', datetime('now'), 'pubmed')",
            (inv2,),
        )
        conn.commit()

        # Duplicate PMID within inv1 - should be rejected
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database) "
                "VALUES (?, '12345', '10.9/dup', 'Paper C', '[]', 'abs', '2022', 'pubmed', 'p3.pdf', 'open_access', datetime('now'), 'pubmed')",
                (inv1,),
            )
            conn.commit()


def test_cross_investigation_relationship_rejected(client):
    """Relationship referencing entity from another investigation is rejected."""
    r = client.post("/investigations", json={"research_question": "Test"})
    assert r.status_code == 201
    inv1 = r.json()["investigation_id"]

    r2 = client.post("/investigations", json={"research_question": "Test2"})
    assert r2.status_code == 201
    inv2 = r2.json()["investigation_id"]

    with get_connection() as conn:
        # Entity in inv1
        conn.execute(
            "INSERT INTO entities (investigation_id, normalized_name, entity_type, stable_identifier, stable_identifier_source) "
            "VALUES (?, 'Drug X', 'drug', 'DB001', 'DrugBank')",
            (inv1,),
        )
        # Entity in inv2
        conn.execute(
            "INSERT INTO entities (investigation_id, normalized_name, entity_type, stable_identifier, stable_identifier_source) "
            "VALUES (?, 'Protein Y', 'protein', 'P12345', 'UniProt')",
            (inv2,),
        )
        conn.commit()

        # Relationship in inv1 referencing entity from inv2 - should be rejected
        with pytest.raises(Exception):
            conn.execute(
                "INSERT INTO relationships (investigation_id, source_entity_id, target_entity_id, relationship_type, confidence_score, extraction_method, source_passage, source_passage_hash, created_at) "
                "VALUES (?, 2, 1, 'inhibits', 0.9, 'rule', 'text', 'hash', datetime('now'))",
                (inv1,),
            )
            conn.commit()


def test_api_crud(client):
    """Investigation CRUD via API."""
    # Create
    r = client.post("/investigations", json={
        "research_question": "Find drug targets for Alzheimer",
        "disease": "Alzheimer's disease",
        "max_papers": 50,
    })
    assert r.status_code == 201
    data = r.json()
    assert data["research_question"] == "Find drug targets for Alzheimer"
    assert data["status"] == "queued"
    assert data["investigation_id"] is not None
    inv_id = data["investigation_id"]

    # List
    r = client.get("/investigations")
    assert r.status_code == 200
    assert len(r.json()) >= 1

    # Get single
    r = client.get(f"/investigations/{inv_id}")
    assert r.status_code == 200
    assert r.json()["investigation_id"] == inv_id

    # Update status
    r = client.patch(f"/investigations/{inv_id}", json={"status": "running"})
    assert r.status_code == 200
    assert r.json()["status"] == "running"

    # Delete
    r = client.delete(f"/investigations/{inv_id}")
    assert r.status_code == 204

    r = client.get(f"/investigations/{inv_id}")
    assert r.status_code == 404


def test_health_endpoint(client):
    """Health endpoint returns ok with foreign_keys enabled."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["foreign_keys"] is True


def test_reset_database():
    """reset_database recreates schema cleanly."""
    reset_database()
    with get_connection() as conn:
        violations = conn.execute("PRAGMA foreign_key_check").fetchall()
        assert len(violations) == 0
