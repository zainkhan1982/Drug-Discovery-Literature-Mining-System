"""M2 acceptance tests: literature acquisition with mocked HTTP, imports, deduplication, limits."""

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Ensure backend app is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import (
    DATA_DIR,
    DATABASE_PATH,
    MAX_PAPERS_PER_INVESTIGATION,
)
from app.database import DATABASE_PATH as DB_PATH2, get_connection, init_database, reset_database
from app.logging import logger
from app.main import create_app
from app.schemas import HealthOut


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create a test client with an isolated temp database."""
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
# PubMed client tests (mocked HTTP)
# -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pubmed_search_mocked():
    """PubMed ESearch returns ID list and count."""
    from app.clients.pubmed import PubMedClient

    client = PubMedClient(base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = lambda: {
        "esearchresult": {
            "count": "42",
            "idlist": ["12345", "67890"],
            "retstart": "0",
            "retmax": "5",
        }
    }

    with patch.object(client, "get", new=AsyncMock(return_value=mock_response)):
        result = await client.search("Alzheimer disease", retmax=5)

    assert result["count"] == 42
    assert result["id_list"] == ["12345", "67890"]
    assert result["retmax"] == 5


@pytest.mark.asyncio
async def test_pubmed_fetch_metadata_mocked():
    """PubMed ESummary returns parsed metadata."""
    from app.clients.pubmed import PubMedClient

    client = PubMedClient(base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = lambda: {
        "result": {
            "12345": {
                "uid": "12345",
                "title": "Test Paper Title",
                "fulljournalname": "Test Journal",
                "pubdate": "2024 Jan",
                "authors": [{"name": "Author One"}, {"name": "Author Two"}],
                "articleids": [{"idtype": "doi", "value": "10.1234/test"}],
            }
        }
    }

    with patch.object(client, "get", new=AsyncMock(return_value=mock_response)):
        results = await client.fetch_metadata(["12345"])

    assert len(results) == 1
    assert results[0]["pmid"] == "12345"
    assert results[0]["title"] == "Test Paper Title"
    assert results[0]["doi"] == "10.1234/test"
    assert results[0]["authors"] == ["Author One", "Author Two"]
    assert results[0]["source"] == "pubmed"


@pytest.mark.asyncio
async def test_pubmed_fetch_abstracts_mocked():
    """PubMed EFetch XML returns abstracts."""
    from app.clients.pubmed import PubMedClient

    client = PubMedClient(base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/")

    xml = """<?xml version="1.0"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <PMID>12345</PMID>
        <Abstract>
          <AbstractText>This is a test abstract about Alzheimer disease.</AbstractText>
        </Abstract>
      </PubmedArticle>
    </PubmedArticleSet>"""

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = xml

    with patch.object(client, "get", new=AsyncMock(return_value=mock_response)):
        abstracts = await client.fetch_abstracts(["12345"])

    assert "12345" in abstracts
    assert "Alzheimer disease" in abstracts["12345"]


# -----------------------------------------------------------------------
# Europe PMC client tests (mocked HTTP)
# -----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_europe_pmc_search_mocked():
    """Europe PMC search returns parsed results."""
    from app.clients.europe_pmc import EuropePMCClient

    client = EuropePMCClient()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = lambda: {
        "hitCount": 100,
        "page": 1,
        "pageSize": 5,
        "resultList": {
            "result": [
                {
                    "pmid": "11111",
                    "pmcid": "PMC12345",
                    "doi": "10.1234/epmc1",
                    "title": "Europe PMC Test Paper",
                    "authorString": "Author A, Author B",
                    "journalTitle": "EPMC Journal",
                    "firstPublicationDate": "2024-06-01",
                    "abstractText": "Test abstract from Europe PMC.",
                    "isOpenAccess": "Y",
                    "citedByCount": "5",
                }
            ]
        },
    }

    with patch.object(client, "get", new=AsyncMock(return_value=mock_response)):
        result = await client.search("Alzheimer", page_size=5)

    assert result["hit_count"] == 100
    assert len(result["papers"]) == 1
    p = result["papers"][0]
    assert p["pmid"] == "11111"
    assert p["pmcid"] == "PMC12345"
    assert p["doi"] == "10.1234/epmc1"
    assert p["title"] == "Europe PMC Test Paper"
    assert p["is_open_access"] is True
    assert p["in_pmc"] is True
    assert p["access_status"] == "open_access_available"
    assert len(p["full_text_urls"]) > 0


@pytest.mark.asyncio
async def test_europe_pmc_not_open_access():
    """Non-open-access papers get abstract_only status."""
    from app.clients.europe_pmc import EuropePMCClient

    client = EuropePMCClient()

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = lambda: {
        "hitCount": 1,
        "page": 1,
        "pageSize": 1,
        "resultList": {
            "result": [
                {
                    "pmid": "22222",
                    "pmcid": None,
                    "doi": None,
                    "title": "Paywalled Paper",
                    "authorString": "Author C",
                    "journalTitle": "Closed Journal",
                    "firstPublicationDate": "2023-01-01",
                    "abstractText": "Paywalled abstract.",
                    "isOpenAccess": "N",
                    "citedByCount": "0",
                }
            ]
        },
    }

    with patch.object(client, "get", new=AsyncMock(return_value=mock_response)):
        result = await client.search("test")

    p = result["papers"][0]
    assert p["is_open_access"] is False
    assert p["in_pmc"] is False
    assert p["access_status"] == "abstract_only"


# -----------------------------------------------------------------------
# Deduplication and metadata normalization
# -----------------------------------------------------------------------
def test_normalize_title():
    """Title normalization for deduplication."""
    from app.services.literature_service import _normalize_title

    assert _normalize_title("Alzheimer's Disease") == "alzheimers disease"
    assert _normalize_title("  Alzheimer   Disease!  ") == "alzheimer disease"
    assert _normalize_title("") == ""
    assert _normalize_title("A-B C_D") == "ab c_d"


def test_deduplicate_by_pmid():
    """Papers with same PMID are deduplicated."""
    from app.services.literature_service import LiteratureService

    service = LiteratureService.__new__(LiteratureService)
    papers = [
        {"pmid": "1", "doi": "10.1/a", "title": "Paper A"},
        {"pmid": "1", "doi": "10.1/b", "title": "Paper A dup"},
        {"pmid": "2", "doi": "10.1/c", "title": "Paper B"},
    ]
    deduped = service._deduplicate_papers(papers)
    assert len(deduped) == 2
    pmid_values = [p["pmid"] for p in deduped]
    assert "1" in pmid_values
    assert "2" in pmid_values


def test_deduplicate_by_doi():
    """Papers with same DOI are deduplicated."""
    from app.services.literature_service import LiteratureService

    service = LiteratureService.__new__(LiteratureService)
    papers = [
        {"pmid": "1", "doi": "10.1/same", "title": "Paper A"},
        {"pmid": "2", "doi": "10.1/same", "title": "Paper B"},
        {"pmid": "3", "doi": "10.1/diff", "title": "Paper C"},
    ]
    deduped = service._deduplicate_papers(papers)
    assert len(deduped) == 2
    doi_values = [p["doi"] for p in deduped if p.get("doi")]
    assert "10.1/same" in doi_values
    assert "10.1/diff" in doi_values


def test_deduplicate_by_title():
    """Papers with same normalized title are deduplicated."""
    from app.services.literature_service import LiteratureService

    service = LiteratureService.__new__(LiteratureService)
    papers = [
        {"pmid": None, "doi": None, "title": "Alzheimer's Disease"},
        {"pmid": None, "doi": None, "title": "alzheimers disease"},
        {"pmid": None, "doi": None, "title": "Different Paper"},
    ]
    deduped = service._deduplicate_papers(papers)
    assert len(deduped) == 2


def test_missing_identifiers_safe():
    """Papers with missing PMID/DOI are handled safely."""
    from app.services.literature_service import LiteratureService

    service = LiteratureService.__new__(LiteratureService)
    papers = [
        {"pmid": None, "doi": None, "title": "No identifiers"},
        {"pmid": None, "doi": None, "title": "Also no identifiers"},
        {"pmid": "1", "doi": None, "title": "Has PMID"},
    ]
    deduped = service._deduplicate_papers(papers)
    # All three kept: two with no identifiers differ by title, one has PMID
    assert len(deduped) == 3


# -----------------------------------------------------------------------
# API: search with mocked clients
# -----------------------------------------------------------------------
def test_api_search_pubmed_mocked(client):
    """Search endpoint works with mocked PubMed client."""
    # Create investigation first
    r = client.post("/investigations", json={"research_question": "Test search"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    # Mock the service methods
    with patch("app.api.literature.get_service") as mock_get_service:
        mock_service = MagicMock()
        mock_service.search_literature = AsyncMock(return_value={
            "papers": [
                {
                    "paper_id": 1,
                    "investigation_id": inv_id,
                    "pmid": "12345",
                    "doi": "10.1234/test",
                    "title": "Mocked Paper",
                    "authors": ["Author A"],
                    "abstract": "Mock abstract.",
                    "publication_date": "2024 Jan",
                    "source": "pubmed",
                    "access_status": "abstract_only",
                    "download_url": None,
                    "full_text_path": None,
                    "retrieved_at": "2024-01-01T00:00:00+00:00",
                    "source_database": "pubmed",
                }
            ],
            "total_found": 1,
            "new_papers": 1,
            "sources": {"pubmed": {"count": 1, "returned": 1}},
            "limit_reached": False,
            "message": None,
        })
        mock_get_service.return_value = mock_service

        r = client.post(
            f"/investigations/{inv_id}/literature/search",
            json={"query": "Alzheimer", "sources": ["pubmed"], "max_papers": 10},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["new_papers"] == 1
        assert data["papers"][0]["title"] == "Mocked Paper"


def test_api_search_invalid_source(client):
    """Search with invalid source returns 400."""
    r = client.post("/investigations", json={"research_question": "Test"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.post(
        f"/investigations/{inv_id}/literature/search",
        json={"query": "test", "sources": ["invalid_source"]},
    )
    assert r.status_code == 400


def test_api_search_nonexistent_investigation(client):
    """Search on nonexistent investigation returns 404."""
    r = client.post(
        "/investigations/99999/literature/search",
        json={"query": "test", "sources": ["pubmed"]},
    )
    assert r.status_code == 404


# -----------------------------------------------------------------------
# API: list papers
# -----------------------------------------------------------------------
def test_api_list_papers(client):
    """List papers endpoint returns paper count and remaining."""
    r = client.post("/investigations", json={"research_question": "Test list"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.get(f"/investigations/{inv_id}/literature")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 0
    assert data["limit"] == 100
    assert data["remaining"] == 100
    assert data["papers"] == []


# -----------------------------------------------------------------------
# API: import PDF/TXT
# -----------------------------------------------------------------------
def test_api_import_txt(client):
    """TXT file import creates a paper record."""
    r = client.post("/investigations", json={"research_question": "Test import"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    txt_content = b"This is a test text file about Alzheimer disease research."
    r = client.post(
        f"/investigations/{inv_id}/literature/import",
        files={"file": ("test.txt", txt_content, "text/plain")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["paper_id"] is not None
    assert data["title"] == "test"
    assert data["access_status"] == "user_imported"
    assert "Alzheimer" in data["extracted_text_preview"]


def test_api_import_pdf(client):
    """PDF file import creates a paper record."""
    r = client.post("/investigations", json={"research_question": "Test import PDF"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    # Create a minimal valid PDF
    pdf_content = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj 2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj 3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000111 00000 n \ntrailer<</Size 4/Root 1 0 R>>startxref\n178\n%%EOF"
    r = client.post(
        f"/investigations/{inv_id}/literature/import",
        files={"file": ("test.pdf", pdf_content, "application/pdf")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is True
    assert data["paper_id"] is not None
    assert data["mime_type"] == "application/pdf"


def test_api_import_invalid_extension(client):
    """Invalid file extension is rejected."""
    r = client.post("/investigations", json={"research_question": "Test invalid"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.post(
        f"/investigations/{inv_id}/literature/import",
        files={"file": ("test.exe", b"binary content", "application/octet-stream")},
    )
    assert r.status_code == 400
    assert "Unsupported file extension" in r.json()["detail"]


def test_api_import_corrupt_pdf(client):
    """Corrupt PDF (bad magic bytes) is rejected."""
    r = client.post("/investigations", json={"research_question": "Test corrupt"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.post(
        f"/investigations/{inv_id}/literature/import",
        files={"file": ("corrupt.pdf", b"not a pdf at all", "application/pdf")},
    )
    assert r.status_code == 400
    assert "Not a valid PDF" in r.json()["detail"]


def test_api_import_nonexistent_investigation(client):
    """Import on nonexistent investigation returns 404."""
    r = client.post(
        "/investigations/99999/literature/import",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 404


# -----------------------------------------------------------------------
# Database: deduplication within investigation
# -----------------------------------------------------------------------
def test_db_dedup_same_pmid_in_one_investigation(client):
    """Same PMID cannot be inserted twice within one investigation."""
    r = client.post("/investigations", json={"research_question": "Test dedup"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    from app.services.literature_service import LiteratureService
    service = LiteratureService()

    # Manually insert first paper
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database, normalized_title, download_url, download_attempted, file_size, mime_type) "
            "VALUES (?, 'PMID1', 'DOI1', 'Paper One', '', '', '2024', 'pubmed', NULL, 'abstract_only', datetime('now'), 'pubmed', 'paper one', NULL, 0, NULL, NULL)",
            (inv_id,),
        )
        conn.commit()

    # Try to insert same PMID again via service
    result = service._store_papers(
        investigation_id=inv_id,
        papers=[
            {
                "pmid": "PMID1",
                "doi": "DOI2",
                "title": "Paper One Dup",
                "authors": [],
                "abstract": "",
                "publication_date": "2024",
                "source_database": "pubmed",
                "source": "pubmed",
                "access_status": "abstract_only",
                "download_url": None,
                "normalized_title": "paper one",
            }
        ],
        limit=100,
    )
    assert len(result) == 0  # Duplicate skipped


def test_db_same_pmid_different_investigation(client):
    """Same PMID can exist in different investigations."""
    r1 = client.post("/investigations", json={"research_question": "Inv 1"})
    r2 = client.post("/investigations", json={"research_question": "Inv 2"})
    inv1 = r1.json()["investigation_id"]
    inv2 = r2.json()["investigation_id"]

    from app.services.literature_service import LiteratureService
    service = LiteratureService()

    # Insert same PMID in both
    result1 = service._store_papers(
        investigation_id=inv1,
        papers=[{"pmid": "SHARED", "doi": None, "title": "Shared Paper", "authors": [], "abstract": "", "publication_date": "2024", "source_database": "pubmed", "source": "pubmed", "access_status": "abstract_only", "download_url": None, "normalized_title": "shared paper"}],
        limit=100,
    )
    result2 = service._store_papers(
        investigation_id=inv2,
        papers=[{"pmid": "SHARED", "doi": None, "title": "Shared Paper", "authors": [], "abstract": "", "publication_date": "2024", "source_database": "pubmed", "source": "pubmed", "access_status": "abstract_only", "download_url": None, "normalized_title": "shared paper"}],
        limit=100,
    )
    assert len(result1) == 1
    assert len(result2) == 1  # Same PMID in different investigation is allowed


# -----------------------------------------------------------------------
# Database: 100-paper limit
# -----------------------------------------------------------------------
def test_db_paper_limit(client):
    """100-paper limit enforced per investigation."""
    r = client.post("/investigations", json={"research_question": "Test limit"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    from app.services.literature_service import LiteratureService
    service = LiteratureService()

    # Insert 100 papers
    papers = []
    for i in range(100):
        papers.append({
            "pmid": f"LIMIT_{i}",
            "doi": None,
            "title": f"Limit Paper {i}",
            "authors": [],
            "abstract": "",
            "publication_date": "2024",
            "source_database": "pubmed",
            "source": "pubmed",
            "access_status": "abstract_only",
            "download_url": None,
            "normalized_title": f"limit paper {i}",
        })
    result = service._store_papers(investigation_id=inv_id, papers=papers, limit=100)
    assert len(result) == 100

    # Try to insert one more - should return 0
    result2 = service._store_papers(
        investigation_id=inv_id,
        papers=[{"pmid": "LIMIT_101", "doi": None, "title": "Over Limit", "authors": [], "abstract": "", "publication_date": "2024", "source_database": "pubmed", "source": "pubmed", "access_status": "abstract_only", "download_url": None, "normalized_title": "over limit"}],
        limit=100,
    )
    assert len(result2) == 0


# -----------------------------------------------------------------------
# API: download endpoint (mocked)
# -----------------------------------------------------------------------
def test_api_download_not_open_access(client):
    """Download on non-open-access paper returns error."""
    r = client.post("/investigations", json={"research_question": "Test download"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    # Insert a non-OA paper manually
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database, normalized_title, download_url, download_attempted, file_size, mime_type) "
            "VALUES (?, 'NOA1', NULL, 'Paywalled Paper', '', '', '2024', 'pubmed', NULL, 'abstract_only', datetime('now'), 'pubmed', 'paywalled paper', NULL, 0, NULL, NULL)",
            (inv_id,),
        )
        conn.commit()
        paper_row = conn.execute("SELECT paper_id FROM papers WHERE pmid = 'NOA1'").fetchone()
        paper_id = paper_row["paper_id"]

    r = client.post(f"/investigations/{inv_id}/literature/{paper_id}/download")
    assert r.status_code == 200
    data = r.json()
    assert data["success"] is False
    assert "not available" in data["error"]


def test_api_download_nonexistent_paper(client):
    """Download on nonexistent paper returns 404."""
    r = client.post("/investigations", json={"research_question": "Test download 404"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    r = client.post(f"/investigations/{inv_id}/literature/99999/download")
    assert r.status_code == 404


# -----------------------------------------------------------------------
# API: delete paper
# -----------------------------------------------------------------------
def test_api_delete_paper(client):
    """Delete paper removes it from investigation."""
    r = client.post("/investigations", json={"research_question": "Test delete"})
    assert r.status_code == 201
    inv_id = r.json()["investigation_id"]

    # Insert a paper
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO papers (investigation_id, pmid, doi, title, authors, abstract, publication_date, source, full_text_path, access_status, retrieved_at, source_database, normalized_title, download_url, download_attempted, file_size, mime_type) "
            "VALUES (?, 'DEL1', NULL, 'To Delete', '', '', '2024', 'pubmed', NULL, 'abstract_only', datetime('now'), 'pubmed', 'to delete', NULL, 0, NULL, NULL)",
            (inv_id,),
        )
        conn.commit()
        paper_row = conn.execute("SELECT paper_id FROM papers WHERE pmid = 'DEL1'").fetchone()
        paper_id = paper_row["paper_id"]

    # Delete it
    r = client.delete(f"/investigations/{inv_id}/literature/{paper_id}")
    assert r.status_code == 204

    # Verify it's gone
    r = client.get(f"/investigations/{inv_id}/literature/{paper_id}")
    assert r.status_code == 404


# -----------------------------------------------------------------------
# Investigation isolation
# -----------------------------------------------------------------------
def test_investigation_isolation(client):
    """Papers from one investigation don't appear in another."""
    r1 = client.post("/investigations", json={"research_question": "Inv A"})
    r2 = client.post("/investigations", json={"research_question": "Inv B"})
    inv1 = r1.json()["investigation_id"]
    inv2 = r2.json()["investigation_id"]

    from app.services.literature_service import LiteratureService
    service = LiteratureService()

    service._store_papers(
        investigation_id=inv1,
        papers=[{"pmid": "ISO1", "doi": None, "title": "Inv A Paper", "authors": [], "abstract": "", "publication_date": "2024", "source_database": "pubmed", "source": "pubmed", "access_status": "abstract_only", "download_url": None, "normalized_title": "inv a paper"}],
        limit=100,
    )

    papers_inv1 = service.list_papers(inv1)
    papers_inv2 = service.list_papers(inv2)
    assert len(papers_inv1) == 1
    assert len(papers_inv2) == 0

