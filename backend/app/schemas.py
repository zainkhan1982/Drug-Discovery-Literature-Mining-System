"""Pydantic schemas for API request/response validation."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class InvestigationCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    research_question: str
    disease: Optional[str] = None
    drug: Optional[str] = None
    gene: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    publication_types: Optional[List[str]] = None
    full_text_preference: Optional[str] = "open_access"
    max_papers: Optional[int] = 100


class InvestigationUpdate(BaseModel):
    status: Optional[str] = None


class InvestigationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    investigation_id: int
    research_question: str
    search_parameters: Dict[str, Any]
    status: str
    created_at: str
    updated_at: str


class HealthOut(BaseModel):
    status: str
    version: str
    database: str
    foreign_keys: bool


class ErrorOut(BaseModel):
    error: str
    detail: Optional[str] = None


# ---- Literature acquisition schemas ----

class LiteratureSearchRequest(BaseModel):
    query: str
    sources: List[str] = ["pubmed", "europepmc"]
    max_papers: Optional[int] = 100
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    publication_types: Optional[List[str]] = None
    open_access_only: Optional[bool] = None


class LiteratureSearchResponse(BaseModel):
    papers: List[Dict[str, Any]]
    total_found: int
    new_papers: int
    sources: Dict[str, Any]
    limit_reached: bool = False
    message: Optional[str] = None


class PaperOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    paper_id: int
    investigation_id: int
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str
    authors: Optional[str] = None
    abstract: Optional[str] = None
    publication_date: Optional[str] = None
    source: str
    full_text_path: Optional[str] = None
    access_status: str
    retrieved_at: str
    source_database: str
    normalized_title: Optional[str] = None
    download_url: Optional[str] = None
    download_attempted: int = 0
    download_error: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None


class PaperListResponse(BaseModel):
    papers: List[PaperOut]
    count: int
    limit: int
    remaining: int


class DownloadResponse(BaseModel):
    success: bool
    paper_id: Optional[int] = None
    error: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None


class ImportRequest(BaseModel):
    pass  # file is uploaded as multipart/form-data


class ImportResponse(BaseModel):
    success: bool
    paper_id: Optional[int] = None
    import_id: Optional[int] = None
    title: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    access_status: Optional[str] = None
    extracted_text_preview: Optional[str] = None
    error: Optional[str] = None


# ---- NLP / biomedical extraction schemas ----

class EntityOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    entity_id: int
    normalized_name: str
    entity_type: str
    external_identifier: Optional[str] = None
    external_identifier_source: Optional[str] = None
    mention_count: int = 0


class EntityMentionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    mention_id: int
    entity_id: int
    paper_id: int
    section_id: Optional[int] = None
    chunk_id: Optional[int] = None
    entity_name_original: str
    page_number: Optional[int] = None


class RelationshipOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    relationship_id: int
    source_entity_id: int
    target_entity_id: int
    relationship_type: str
    confidence_score: float
    extraction_method: str
    is_direct_interaction: int
    negation: int
    uncertainty: int
    speculation: int
    experimental_context: Optional[str] = None
    source_passage: str
    source_passage_hash: str
    created_at: str
    source_name: Optional[str] = None
    target_name: Optional[str] = None


class EvidenceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    evidence_id: int
    relationship_id: int
    paper_id: int
    section_id: Optional[int] = None
    supporting_text: str
    page_number: Optional[int] = None
    evidence_type: Optional[str] = None
    extraction_method: str
    confidence_score: float
    extracted_at: str


class ProcessPaperRequest(BaseModel):
    paper_id: int


class ProcessPaperResponse(BaseModel):
    success: bool
    paper_id: Optional[int] = None
    processing_status: Optional[str] = None
    entities_extracted: int = 0
    relationships_extracted: int = 0
    sections_found: int = 0
    chunks_created: int = 0
    error: Optional[str] = None


class ProcessAllResponse(BaseModel):
    investigation_id: int
    total_papers: int
    processed: int
    skipped: int
    failed: int
    paper_results: list[dict[str, Any]]


class ProcessingStatusOut(BaseModel):
    paper_id: int
    processing_status: str
    download_error: Optional[str] = None


class ReprocessResponse(BaseModel):
    success: bool
    paper_id: Optional[int] = None
    processing_status: Optional[str] = None
    entities_extracted: int = 0
    relationships_extracted: int = 0
    error: Optional[str] = None
