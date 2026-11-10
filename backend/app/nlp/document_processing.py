"""Document processing: PDF/TXT/XML extraction, section detection, chunking."""

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, MAX_CHUNKS_PER_PAPER
from app.logging import logger


SECTION_PATTERNS = [
    (r"^\s*abstract\s*$", "abstract", 0),
    (r"^\s*introduction\s*$", "introduction", 1),
    (r"^\s*methods\s*$", "methods", 2),
    (r"^\s*materials?\s+and\s+methods\s*$", "methods", 2),
    (r"^\s*results\s*$", "results", 3),
    (r"^\s*discussion\s*$", "discussion", 4),
    (r"^\s*conclusion\s*$", "conclusion", 5),
    (r"^\s*references\s*$", "references", 6),
]


@dataclass
class ExtractedSection:
    section_name: str
    section_order: int
    text: str
    page_start: int
    page_end: int


@dataclass
class ExtractedChunk:
    chunk_text: str
    chunk_order: int
    section_id: int | None
    page_start: int
    page_end: int
    token_count: int


def extract_pdf_text(file_path: str) -> tuple[str, list[ExtractedSection], list[ExtractedChunk]]:
    """Extract text from PDF using PyMuPDF with section detection and chunking.

    Returns (full_text, sections, chunks).
    """
    doc = fitz.open(file_path)
    full_text_parts: list[str] = []
    sections: list[ExtractedSection] = []
    chunks: list[ExtractedChunk] = []

    current_section: ExtractedSection | None = None
    section_counter = 0
    chunk_counter = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():
            continue

        page_text_lines = text.split("\n")
        page_has_section_header = False

        # Check for section headers on this page
        for line in page_text_lines:
            stripped = line.strip()
            for pattern, section_name, order in SECTION_PATTERNS:
                if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 80:
                    # Save previous section
                    if current_section is not None:
                        sections.append(current_section)
                    current_section = ExtractedSection(
                        section_name=section_name,
                        section_order=order,
                        text="",
                        page_start=page_num + 1,
                        page_end=page_num + 1,
                    )
                    section_counter += 1
                    page_has_section_header = True
                    break
            if page_has_section_header:
                break

        # Add text to current section
        if current_section is not None:
            current_section.text += f"\n{text}"
            current_section.page_end = page_num + 1
        else:
            # Text before any section header - assign to "abstract" by default
            if current_section is None:
                current_section = ExtractedSection(
                    section_name="abstract",
                    section_order=0,
                    text=text,
                    page_start=page_num + 1,
                    page_end=page_num + 1,
                )
                section_counter += 1

        full_text_parts.append(f"\n--- Page {page_num + 1} ---\n{text}")

    # Save last section
    if current_section is not None:
        sections.append(current_section)

    doc.close()

    full_text = "\n".join(full_text_parts)

    # Create chunks from sections
    for section in sections:
        section_chunks = _chunk_text(
            section.text,
            chunk_counter,
            section.section_id if hasattr(section, "section_id") else None,
            section.page_start,
            section.page_end,
        )
        for chunk in section_chunks:
            chunks.append(chunk)
            chunk_counter += 1
            if chunk_counter >= MAX_CHUNKS_PER_PAPER:
                break
        if chunk_counter >= MAX_CHUNKS_PER_PAPER:
            break

    logger.info(
        "Extracted %d sections, %d chunks from %s",
        len(sections), len(chunks), file_path,
    )

    return full_text, sections, chunks


def _chunk_text(
    text: str,
    start_order: int,
    section_id: int | None,
    page_start: int,
    page_end: int,
) -> list[ExtractedChunk]:
    """Split text into overlapping chunks."""
    if not text.strip():
        return []

    # Simple word-based chunking
    words = text.split()
    chunks: list[ExtractedChunk] = []
    order = start_order
    i = 0

    while i < len(words) and len(chunks) < MAX_CHUNKS_PER_PAPER:
        # Take a chunk of CHUNK_SIZE words
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)

        chunks.append(
            ExtractedChunk(
                chunk_text=chunk_text,
                chunk_order=order,
                section_id=section_id,
                page_start=page_start,
                page_end=page_end,
                token_count=len(chunk_words),
            )
        )
        order += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP

        if i >= len(words):
            break

    return chunks


def extract_txt_text(file_path: str) -> tuple[str, list[ExtractedSection], list[ExtractedChunk]]:
    """Extract text from TXT file with basic section detection and chunking."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Try to detect sections
    sections: list[ExtractedSection] = []
    current_section: ExtractedSection | None = None
    section_counter = 0

    for line in text.split("\n"):
        stripped = line.strip()
        for pattern, section_name, order in SECTION_PATTERNS:
            if re.match(pattern, stripped, re.IGNORECASE) and len(stripped) < 80:
                if current_section is not None:
                    sections.append(current_section)
                current_section = ExtractedSection(
                    section_name=section_name,
                    section_order=order,
                    text="",
                    page_start=1,
                    page_end=1,
                )
                section_counter += 1
                break
        if current_section is not None:
            current_section.text += f"\n{line}"

    if current_section is not None:
        sections.append(current_section)

    if not sections:
        sections.append(
            ExtractedSection(
                section_name="abstract",
                section_order=0,
                text=text,
                page_start=1,
                page_end=1,
            )
        )

    # Create chunks
    chunks: list[ExtractedChunk] = []
    chunk_counter = 0
    for section in sections:
        section_chunks = _chunk_text(
            section.text,
            chunk_counter,
            None,
            1,
            1,
        )
        for chunk in section_chunks:
            chunks.append(chunk)
            chunk_counter += 1
            if chunk_counter >= MAX_CHUNKS_PER_PAPER:
                break
        if chunk_counter >= MAX_CHUNKS_PER_PAPER:
            break

    return text, sections, chunks


def extract_xml_text(file_path: str) -> tuple[str, list[ExtractedSection], list[ExtractedChunk]]:
    """Extract text from Europe PMC full-text XML."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract article title
    title_elem = root.find(".//article-title")
    title = "".join(title_elem.itertext()) if title_elem is not None else ""

    # Extract abstract
    abstract_parts: list[str] = []
    for abs_elem in root.findall(".//abstract"):
        for text_elem in abs_elem.findall(".//"):
            t = "".join(text_elem.itertext()).strip()
            if t:
                abstract_parts.append(t)
    abstract_text = "\n".join(abstract_parts)

    # Extract body sections
    sections: list[ExtractedSection] = []
    section_counter = 0

    for sec in root.findall(".//body//sec"):
        title_elem = sec.find("title")
        sec_title = "".join(title_elem.itertext()) if title_elem is not None else "section"
        sec_text_parts: list[str] = []
        for p in sec.findall(".//p"):
            p_text = "".join(p.itertext()).strip()
            if p_text:
                sec_text_parts.append(p_text)
        sec_text = "\n".join(sec_text_parts)

        sections.append(
            ExtractedSection(
                section_name=sec_title.lower(),
                section_order=section_counter,
                text=sec_text,
                page_start=1,
                page_end=1,
            )
        )
        section_counter += 1

    full_text = f"Title: {title}\nAbstract: {abstract_text}\n\n"
    for s in sections:
        full_text += f"\n## {s.section_name}\n{s.text}"

    # Create chunks
    chunks: list[ExtractedChunk] = []
    chunk_counter = 0
    for section in sections:
        section_chunks = _chunk_text(
            section.text,
            chunk_counter,
            None,
            1,
            1,
        )
        for chunk in section_chunks:
            chunks.append(chunk)
            chunk_counter += 1
            if chunk_counter >= MAX_CHUNKS_PER_PAPER:
                break
        if chunk_counter >= MAX_CHUNKS_PER_PAPER:
            break

    return full_text, sections, chunks


def compute_passage_hash(passage: str) -> str:
    """Compute SHA-256 hash of a passage for provenance verification."""
    return hashlib.sha256(passage.encode("utf-8")).hexdigest()


def clean_text(text: str) -> str:
    """Clean text without destroying scientific meaning.

    - Normalize whitespace
    - Remove control characters
    - Preserve hyphenated words, numbers, units
    """
    if not text:
        return ""
    # Replace control characters with space
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
