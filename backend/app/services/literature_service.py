"""Literature acquisition service layer.

Orchestrates searches across PubMed and Europe PMC, deduplication,
database storage, full-text downloads, and local imports.
"""

import hashlib
import magic
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import httpx

from app.clients.europe_pmc import EuropePMCClient
from app.clients.pubmed import PubMedClient
from app.config import (
    EUROPEPMC_BASE_URL,
    FULL_TEXT_DIR,
    MAX_IMPORT_FILE_SIZE_MB,
    MAX_PAPERS_PER_INVESTIGATION,
    PUBMED_BASE_URL,
    PUBMED_EMAIL,
    PUBMED_RATE_LIMIT,
)
from app.database import get_connection, get_transaction
from app.logging import logger


def _normalize_title(title: str) -> str:
    """Normalize a title for deduplication: lowercase, strip punctuation, collapse whitespace."""
    if not title:
        return ""
    t = title.lower()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _parse_authors(author_string: str | list[str]) -> list[str]:
    """Parse author string into list. Europe PMC returns comma-separated string."""
    if isinstance(author_string, list):
        return [a for a in author_string if a]
    if isinstance(author_string, str):
        return [a.strip() for a in author_string.split(",") if a.strip()]
    return []


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LiteratureService:
    """Service for literature discovery, acquisition, and management."""

    def __init__(self):
        self.pubmed = PubMedClient(
            base_url=PUBMED_BASE_URL,
            email=PUBMED_EMAIL,
            rate_limit=PUBMED_RATE_LIMIT,
        )
        self.europe_pmc = EuropePMCClient(base_url=EUROPEPMC_BASE_URL)
        self._http_client = httpx.AsyncClient(timeout=60.0)

    async def close(self) -> None:
        await self.pubmed.close()
        await self.europe_pmc.close()
        await self._http_client.aclose()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    async def search_literature(
        self,
        investigation_id: int,
        query: str,
        sources: list[str],
        max_papers: int = 100,
        date_range: dict | None = None,
        publication_types: list[str] | None = None,
        open_access_only: bool | None = None,
    ) -> dict[str, Any]:
        """Search literature sources and store new papers.

        Returns merged results with deduplication applied.
        """
        all_papers: list[dict[str, Any]] = []
        source_results: dict[str, dict[str, Any]] = {}

        # Enforce limit transactionally
        current_count = self._count_papers(investigation_id)
        remaining = MAX_PAPERS_PER_INVESTIGATION - current_count
        if remaining <= 0:
            return {
                "papers": [],
                "total_found": 0,
                "new_papers": 0,
                "sources": source_results,
                "limit_reached": True,
                "message": f"Investigation already at {MAX_PAPERS_PER_INVESTIGATION} paper limit",
            }

        effective_limit = min(max_papers, remaining, MAX_PAPERS_PER_INVESTIGATION)

        # Search PubMed
        if "pubmed" in sources:
            try:
                pm_result = await self.pubmed.search_and_fetch(
                    term=query,
                    retmax=effective_limit,
                    date_range=date_range,
                    publication_types=publication_types,
                    fetch_abstracts=True,
                )
                for p in pm_result.get("papers", []):
                    p["source_database"] = "pubmed"
                    p["normalized_title"] = _normalize_title(p.get("title", ""))
                all_papers.extend(pm_result.get("papers", []))
                source_results["pubmed"] = {
                    "count": pm_result.get("count", 0),
                    "returned": len(pm_result.get("papers", [])),
                }
                logger.info(
                    "PubMed search returned %d papers (total %d)",
                    len(pm_result.get("papers", [])),
                    pm_result.get("count", 0),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("PubMed search failed: %s", exc)
                source_results["pubmed"] = {"error": str(exc)}

        # Search Europe PMC
        if "europepmc" in sources:
            try:
                ep_result = await self.europe_pmc.search_and_fetch(
                    query=query,
                    page_size=effective_limit,
                    date_range=date_range,
                    publication_types=publication_types,
                    open_access=open_access_only,
                )
                for p in ep_result.get("papers", []):
                    p["source_database"] = "europepmc"
                    p["normalized_title"] = _normalize_title(p.get("title", ""))
                all_papers.extend(ep_result.get("papers", []))
                source_results["europepmc"] = {
                    "hit_count": ep_result.get("hit_count", 0),
                    "returned": len(ep_result.get("papers", [])),
                }
                logger.info(
                    "Europe PMC search returned %d papers (total %d)",
                    len(ep_result.get("papers", [])),
                    ep_result.get("hit_count", 0),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Europe PMC search failed: %s", exc)
                source_results["europepmc"] = {"error": str(exc)}

        # Deduplicate within this search batch
        deduped = self._deduplicate_papers(all_papers)

        # Store new papers in database
        stored = self._store_papers(investigation_id, deduped, effective_limit)

        return {
            "papers": stored,
            "total_found": sum(
                s.get("count", s.get("hit_count", 0)) for s in source_results.values() if isinstance(s, dict)
            ),
            "new_papers": len(stored),
            "sources": source_results,
            "limit_reached": len(stored) < len(deduped),
            "message": None,
        }

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    def _deduplicate_papers(
        self, papers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Deduplicate by PMID first, then DOI, then normalized title."""
        seen_pmid: set[str] = set()
        seen_doi: set[str] = set()
        seen_title: set[str] = set()
        deduped: list[dict[str, Any]] = []

        for p in papers:
            pmid = p.get("pmid", "") or ""
            doi = p.get("doi", "") or ""
            norm_title = p.get("normalized_title") or _normalize_title(p.get("title", ""))

            # Skip if PMID already seen (and PMID is not empty)
            if pmid and pmid in seen_pmid:
                continue
            # Skip if DOI already seen (and DOI is not empty)
            if doi and doi in seen_doi:
                continue
            # Skip if normalized title already seen (and title is not empty)
            if norm_title and norm_title in seen_title:
                continue

            seen_pmid.add(pmid)
            seen_doi.add(doi)
            seen_title.add(norm_title)
            deduped.append(p)

        return deduped

    # ------------------------------------------------------------------
    # Database storage
    # ------------------------------------------------------------------
    def _count_papers(self, investigation_id: int) -> int:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM papers WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchone()
            return row[0] if row else 0

    def _store_papers(
        self,
        investigation_id: int,
        papers: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Store papers in database, enforcing per-investigation limit and dedup against existing."""
        stored: list[dict[str, Any]] = []

        with get_transaction() as conn:
            current_count = self._count_papers(investigation_id)
            remaining = MAX_PAPERS_PER_INVESTIGATION - current_count
            if remaining <= 0:
                return stored

            # Get existing PMIDs/DOIs for this investigation
            existing_rows = conn.execute(
                "SELECT pmid, doi, normalized_title FROM papers WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchall()
            existing_pmid = {r["pmid"] for r in existing_rows if r["pmid"]}
            existing_doi = {r["doi"] for r in existing_rows if r["doi"]}
            existing_title = {r["normalized_title"] for r in existing_rows if r["normalized_title"]}

            now = _iso_now()

            for p in papers:
                if len(stored) >= min(limit, remaining):
                    break

                pmid = p.get("pmid", "") or ""
                doi = p.get("doi", "") or ""
                norm_title = p.get("normalized_title") or _normalize_title(p.get("title", ""))

                # Skip if already in DB
                if pmid and pmid in existing_pmid:
                    continue
                if doi and doi in existing_doi:
                    continue
                if norm_title and norm_title in existing_title:
                    continue

                # Determine access status
                access_status = p.get("access_status", "abstract_only")
                if p.get("is_open_access") and p.get("in_pmc"):
                    access_status = "open_access_available"

                # Determine source
                source = p.get("source_database", p.get("source", "unknown"))

                # Build full_text URL
                download_url = None
                full_text_urls = p.get("full_text_urls", [])
                if full_text_urls:
                    download_url = full_text_urls[0]

                authors = p.get("authors", [])
                if isinstance(authors, str):
                    authors = [a.strip() for a in authors.split(",") if a.strip()]
                elif not isinstance(authors, list):
                    authors = []

                try:
                    cur = conn.execute(
                        """
                        INSERT INTO papers
                        (investigation_id, pmid, doi, title, authors, abstract,
                         publication_date, source, full_text_path, access_status,
                         retrieved_at, source_database, normalized_title,
                         download_url, download_attempted, file_size, mime_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            investigation_id,
                            pmid or None,
                            doi or None,
                            p.get("title", ""),
                            ",".join(authors) if authors else "",
                            p.get("abstract", ""),
                            p.get("publication_date", ""),
                            source,
                            None,  # full_text_path - set after download
                            access_status,
                            now,
                            source,
                            norm_title or None,
                            download_url,
                            0,  # download_attempted
                            None,  # file_size
                            None,  # mime_type
                        ),
                    )
                    paper_id = cur.lastrowid

                    # Update tracking sets
                    if pmid:
                        existing_pmid.add(pmid)
                    if doi:
                        existing_doi.add(doi)
                    if norm_title:
                        existing_title.add(norm_title)

                    stored.append(
                        {
                            "paper_id": paper_id,
                            "investigation_id": investigation_id,
                            "pmid": pmid,
                            "doi": doi,
                            "title": p.get("title", ""),
                            "authors": authors,
                            "abstract": p.get("abstract", ""),
                            "publication_date": p.get("publication_date", ""),
                            "source": source,
                            "access_status": access_status,
                            "download_url": download_url,
                            "full_text_path": None,
                            "retrieved_at": now,
                            "source_database": source,
                        }
                    )
                    logger.info(
                        "Stored paper %d (%s) for investigation %d",
                        paper_id,
                        pmid or doi or norm_title,
                        investigation_id,
                    )
                except sqlite3.IntegrityError as exc:
                    logger.warning(
                        "Skipped duplicate paper (%s): %s",
                        pmid or doi or norm_title,
                        exc,
                    )
                    continue

        return stored

    # ------------------------------------------------------------------
    # List papers
    # ------------------------------------------------------------------
    def list_papers(self, investigation_id: int) -> list[dict[str, Any]]:
        """List all papers for an investigation."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT paper_id, investigation_id, pmid, doi, title, authors, abstract,
                       publication_date, source, full_text_path, access_status,
                       retrieved_at, source_database, normalized_title,
                       download_url, download_attempted, download_error,
                       file_size, mime_type
                FROM papers WHERE investigation_id = ?
                ORDER BY retrieved_at DESC
                """,
                (investigation_id,),
            ).fetchall()

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Get paper
    # ------------------------------------------------------------------
    def get_paper(self, investigation_id: int, paper_id: int) -> dict[str, Any] | None:
        """Get a single paper by ID, ensuring it belongs to the investigation."""
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT paper_id, investigation_id, pmid, doi, title, authors, abstract,
                       publication_date, source, full_text_path, access_status,
                       retrieved_at, source_database, normalized_title,
                       download_url, download_attempted, download_error,
                       file_size, mime_type, processing_status
                FROM papers
                WHERE paper_id = ? AND investigation_id = ?
                """,
                (paper_id, investigation_id),
            ).fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # Download full text
    # ------------------------------------------------------------------
    async def download_full_text(
        self, investigation_id: int, paper_id: int
    ) -> dict[str, Any]:
        """Download full text for a paper if open access is available."""
        paper = self.get_paper(investigation_id, paper_id)
        if not paper:
            return {"success": False, "error": "Paper not found"}

        if paper["access_status"] not in ("open_access_available",):
            return {
                "success": False,
                "error": f"Full text not available (status: {paper['access_status']})",
            }

        download_url = paper.get("download_url")
        if not download_url:
            # Try to construct from PMCID
            # We don't have PMCID stored directly; try DOI URL
            if paper.get("doi"):
                download_url = f"https://doi.org/{paper['doi']}"
            else:
                return {
                    "success": False,
                    "error": "No download URL available",
                }

        return await self._download_and_store(
            investigation_id, paper_id, download_url
        )

    async def _download_and_store(
        self, investigation_id: int, paper_id: int, url: str
    ) -> dict[str, Any]:
        """Download content from URL and store it."""
        now = _iso_now()

        # Log the attempt
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO paper_download_logs
                (investigation_id, paper_id, url, attempt_at, success, status_code, error_message)
                VALUES (?, ?, ?, ?, 0, NULL, NULL)
                """,
                (investigation_id, paper_id, url, now),
            )
            log_id = conn.execute(
                "SELECT last_insert_rowid()"
            ).fetchone()[0]
            conn.commit()

        try:
            # Mark download attempted
            with get_connection() as conn:
                conn.execute(
                    "UPDATE papers SET download_attempted = 1 WHERE paper_id = ?",
                    (paper_id,),
                )
                conn.commit()

            response = await self._http_client.get(url, follow_redirects=True)
            status_code = response.status_code

            if status_code != 200:
                error_msg = f"HTTP {status_code}"
                self._log_download_failure(investigation_id, paper_id, log_id, status_code, error_msg)
                self._mark_download_failed(paper_id, error_msg)
                return {"success": False, "error": error_msg}

            content = response.content
            content_type = response.headers.get("content-type", "")

            # Determine file extension
            if "xml" in content_type.lower() or url.endswith(".xml"):
                ext = ".xml"
            elif "pdf" in content_type.lower():
                ext = ".pdf"
            else:
                ext = ".xml"  # default for Europe PMC full text

            # Safe filename
            safe_name = f"paper_{paper_id}{ext}"
            stored_path = FULL_TEXT_DIR / safe_name

            # Avoid collisions
            counter = 1
            while stored_path.exists():
                stored_path = FULL_TEXT_DIR / f"paper_{paper_id}_{counter}{ext}"
                counter += 1

            # Write file
            stored_path.write_bytes(content)
            file_size = len(content)

            # Detect MIME type using magic
            mime = magic.from_file(str(stored_path), mime=True)

            # Update paper record
            relative_path = str(stored_path.relative_to(Path(__file__).resolve().parent.parent.parent))
            with get_transaction() as conn:
                conn.execute(
                    """
                    UPDATE papers
                    SET full_text_path = ?, access_status = 'downloaded',
                        file_size = ?, mime_type = ?
                    WHERE paper_id = ?
                    """,
                    (relative_path, file_size, mime, paper_id),
                )
                conn.execute(
                    """
                    UPDATE paper_download_logs
                    SET success = 1, status_code = ?, file_size = ?
                    WHERE log_id = ?
                    """,
                    (status_code, file_size, log_id),
                )
                conn.commit()

            logger.info(
                "Downloaded full text for paper %d: %s (%d bytes, %s)",
                paper_id, stored_path.name, file_size, mime,
            )

            return {
                "success": True,
                "paper_id": paper_id,
                "file_path": relative_path,
                "file_size": file_size,
                "mime_type": mime,
            }

        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            logger.error("Download failed for paper %d: %s", paper_id, error_msg)
            self._log_download_failure(investigation_id, paper_id, log_id, None, error_msg)
            self._mark_download_failed(paper_id, error_msg)
            return {"success": False, "error": error_msg}

    def _log_download_failure(
        self, investigation_id: int, paper_id: int, log_id: int,
        status_code: int | None, error_msg: str
    ) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE paper_download_logs
                SET success = 0, status_code = ?, error_message = ?
                WHERE log_id = ?
                """,
                (status_code, error_msg, log_id),
            )
            conn.commit()

    def _mark_download_failed(self, paper_id: int, error_msg: str) -> None:
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE papers
                SET access_status = 'download_failed', download_error = ?
                WHERE paper_id = ?
                """,
                (error_msg, paper_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Local import (PDF/TXT)
    # ------------------------------------------------------------------
    def import_file(
        self, investigation_id: int, file_bytes: bytes, original_filename: str
    ) -> dict[str, Any]:
        """Import a user-uploaded PDF or TXT file."""
        # Validate extension
        ext = Path(original_filename).suffix.lower()
        allowed = [e.strip() for e in os.getenv("ALLOWED_IMPORT_EXTENSIONS", ".pdf,.txt").split(",")]
        if ext not in allowed:
            return {
                "success": False,
                "error": f"Unsupported file extension: {ext}. Allowed: {', '.join(allowed)}",
            }

        # Validate size
        max_bytes = MAX_IMPORT_FILE_SIZE_MB * 1024 * 1024
        if len(file_bytes) > max_bytes:
            return {
                "success": False,
                "error": f"File too large: {len(file_bytes)} bytes (max {MAX_IMPORT_FILE_SIZE_MB} MB)",
            }

        # Validate magic bytes
        mime = magic.from_buffer(file_bytes, mime=True)
        if ext == ".pdf" and not file_bytes.startswith(b"%PDF"):
            return {"success": False, "error": "Not a valid PDF file (bad magic bytes)"}
        if ext == ".txt":
            # Try to decode as text
            try:
                file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_bytes.decode("latin-1")
                except UnicodeDecodeError:
                    return {"success": False, "error": "Not a valid text file"}

        # Generate safe filename
        safe_name = self._safe_filename(original_filename)
        stored_path = FULL_TEXT_DIR / safe_name

        # Avoid collisions
        counter = 1
        while stored_path.exists():
            stem = Path(safe_name).stem
            suffix = Path(safe_name).suffix
            stored_path = FULL_TEXT_DIR / f"{stem}_{counter}{suffix}"
            counter += 1

        # Write file
        stored_path.write_bytes(file_bytes)
        file_size = len(file_bytes)

        # Extract text
        extracted_text = ""
        section_info = None
        if ext == ".pdf":
            try:
                extracted_text, section_info = self._extract_pdf_text(str(stored_path))
            except Exception as exc:  # noqa: BLE001
                logger.warning("PDF text extraction failed for %s: %s", original_filename, exc)
                extracted_text = ""
        else:
            try:
                extracted_text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                extracted_text = file_bytes.decode("latin-1")

        # Store import record
        now = _iso_now()
        with get_transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO paper_imports
                (investigation_id, original_filename, stored_filename, stored_path,
                 file_size, mime_type, import_status, imported_at)
                VALUES (?, ?, ?, ?, ?, ?, 'processed', ?)
                """,
                (
                    investigation_id,
                    original_filename,
                    stored_path.name,
                    str(stored_path.relative_to(Path(__file__).resolve().parent.parent.parent)),
                    file_size,
                    mime,
                    now,
                ),
            )
            import_id = cur.lastrowid

            # Create paper record from import
            title = Path(original_filename).stem
            cur2 = conn.execute(
                """
                INSERT INTO papers
                (investigation_id, pmid, doi, title, authors, abstract,
                 publication_date, source, full_text_path, access_status,
                 retrieved_at, source_database, normalized_title,
                 download_url, download_attempted, file_size, mime_type)
                VALUES (?, NULL, NULL, ?, ?, ?, NULL, 'user_import', ?, 'user_imported',
                        ?, 'user_import', ?, NULL, 0, ?, ?)
                """,
                (
                    investigation_id,
                    title,
                    "",  # authors
                    extracted_text[:2000],  # abstract preview
                    str(stored_path.relative_to(Path(__file__).resolve().parent.parent.parent)),
                    now,
                    _normalize_title(title),
                    file_size,
                    mime,
                ),
            )
            paper_id = cur2.lastrowid

            # Link import to paper
            conn.execute(
                "UPDATE paper_imports SET paper_id = ? WHERE import_id = ?",
                (paper_id, import_id),
            )
            conn.commit()

        logger.info(
            "Imported file %s as paper %d for investigation %d",
            original_filename, paper_id, investigation_id,
        )

        return {
            "success": True,
            "paper_id": paper_id,
            "import_id": import_id,
            "title": title,
            "file_size": file_size,
            "mime_type": mime,
            "access_status": "user_imported",
            "extracted_text_preview": extracted_text[:500],
        }

    def _safe_filename(self, filename: str) -> str:
        """Generate a safe filesystem filename from user-provided name."""
        # Remove path components
        name = os.path.basename(filename)
        # Replace unsafe characters
        name = re.sub(r"[^\w\s\.-]", "_", name)
        # Limit length
        if len(name) > 100:
            stem = Path(name).stem[:80]
            suffix = Path(name).suffix
            name = f"{stem}{suffix}"
        return name

    def _extract_pdf_text(self, file_path: str) -> tuple[str, dict | None]:
        """Extract text from PDF using PyMuPDF. Returns (text, section_info)."""
        doc = fitz.open(file_path)
        parts: list[str] = []
        sections: list[dict] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                parts.append(f"\n--- Page {page_num + 1} ---\n{text}")
                sections.append(
                    {
                        "page": page_num + 1,
                        "text_preview": text[:200],
                    }
                )

        doc.close()
        return "\n".join(parts), {"sections": sections} if sections else None

    # ------------------------------------------------------------------
    # Delete paper
    # ------------------------------------------------------------------
    def delete_paper(self, investigation_id: int, paper_id: int) -> bool:
        """Delete a paper and its associated files."""
        paper = self.get_paper(investigation_id, paper_id)
        if not paper:
            return False

        # Delete full text file if exists
        if paper.get("full_text_path"):
            try:
                fpath = Path(__file__).resolve().parent.parent.parent / paper["full_text_path"]
                if fpath.exists():
                    fpath.unlink()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to delete file %s: %s", paper["full_text_path"], exc)

        with get_transaction() as conn:
            conn.execute(
                "DELETE FROM papers WHERE paper_id = ? AND investigation_id = ?",
                (paper_id, investigation_id),
            )
            conn.commit()

        logger.info("Deleted paper %d from investigation %d", paper_id, investigation_id)
        return True

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------
    def get_paper_count(self, investigation_id: int) -> int:
        return self._count_papers(investigation_id)


import sqlite3  # noqa: E402  (needed by _store_papers)
