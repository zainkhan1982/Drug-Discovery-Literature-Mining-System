-- ============================================================================
-- Migration V001: Initial schema for Drug Discovery Literature Mining System
-- Investigation-level isolation enforced via composite foreign keys
-- ============================================================================

CREATE TABLE IF NOT EXISTS investigations (
    investigation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    research_question TEXT NOT NULL,
    search_parameters TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    pmid TEXT,
    doi TEXT,
    title TEXT NOT NULL,
    authors TEXT,
    abstract TEXT,
    publication_date TEXT,
    source TEXT NOT NULL,
    full_text_path TEXT,
    access_status TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    source_database TEXT NOT NULL,
    UNIQUE (paper_id, investigation_id),
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_pmid_per_inv ON papers(pmid, investigation_id) WHERE pmid IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_papers_doi_per_inv ON papers(doi, investigation_id) WHERE doi IS NOT NULL;

CREATE TABLE IF NOT EXISTS paper_sections (
    section_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    investigation_id INTEGER NOT NULL,
    section_name TEXT NOT NULL,
    section_order INTEGER,
    page_start INTEGER,
    page_end INTEGER,
    UNIQUE (section_id, paper_id, investigation_id),
    FOREIGN KEY (paper_id, investigation_id) REFERENCES papers(paper_id, investigation_id)
);

CREATE TABLE IF NOT EXISTS paper_chunks (
    chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    investigation_id INTEGER NOT NULL,
    section_id INTEGER,
    chunk_text TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    page_start INTEGER,
    page_end INTEGER,
    token_count INTEGER,
    UNIQUE (chunk_id, paper_id, investigation_id),
    FOREIGN KEY (paper_id, investigation_id) REFERENCES papers(paper_id, investigation_id),
    FOREIGN KEY (section_id, paper_id, investigation_id) REFERENCES paper_sections(section_id, paper_id, investigation_id)
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    normalized_name TEXT,
    entity_type TEXT NOT NULL,
    stable_identifier TEXT,
    stable_identifier_source TEXT,
    UNIQUE (entity_id, investigation_id),
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_normalized_per_inv ON entities(normalized_name, investigation_id) WHERE normalized_name IS NOT NULL;

CREATE TABLE IF NOT EXISTS entity_synonyms (
    synonym_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id INTEGER NOT NULL,
    investigation_id INTEGER NOT NULL,
    synonym TEXT NOT NULL,
    source TEXT,
    UNIQUE (synonym_id, entity_id, investigation_id),
    FOREIGN KEY (entity_id, investigation_id) REFERENCES entities(entity_id, investigation_id)
);

CREATE TABLE IF NOT EXISTS paper_entity_mentions (
    mention_id INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id INTEGER NOT NULL,
    investigation_id INTEGER NOT NULL,
    section_id INTEGER,
    entity_id INTEGER NOT NULL,
    chunk_id INTEGER,
    entity_name_original TEXT NOT NULL,
    page_number INTEGER,
    UNIQUE (mention_id, paper_id, investigation_id, entity_id),
    FOREIGN KEY (paper_id, investigation_id) REFERENCES papers(paper_id, investigation_id),
    FOREIGN KEY (entity_id, investigation_id) REFERENCES entities(entity_id, investigation_id),
    FOREIGN KEY (section_id, paper_id, investigation_id) REFERENCES paper_sections(section_id, paper_id, investigation_id),
    FOREIGN KEY (chunk_id, paper_id, investigation_id) REFERENCES paper_chunks(chunk_id, paper_id, investigation_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    relationship_id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    source_entity_id INTEGER NOT NULL,
    target_entity_id INTEGER NOT NULL,
    relationship_type TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    extraction_method TEXT NOT NULL,
    is_direct_interaction INTEGER NOT NULL DEFAULT 0,
    negation INTEGER NOT NULL DEFAULT 0,
    uncertainty INTEGER NOT NULL DEFAULT 0,
    speculation INTEGER NOT NULL DEFAULT 0,
    experimental_context TEXT,
    source_passage TEXT NOT NULL,
    source_passage_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (relationship_id, investigation_id),
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id),
    FOREIGN KEY (source_entity_id, investigation_id) REFERENCES entities(entity_id, investigation_id),
    FOREIGN KEY (target_entity_id, investigation_id) REFERENCES entities(entity_id, investigation_id)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_id INTEGER NOT NULL,
    investigation_id INTEGER NOT NULL,
    paper_id INTEGER NOT NULL,
    section_id INTEGER,
    supporting_text TEXT NOT NULL,
    page_number INTEGER,
    evidence_type TEXT,
    extraction_method TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    extracted_at TEXT NOT NULL,
    UNIQUE (evidence_id, investigation_id),
    FOREIGN KEY (relationship_id, investigation_id) REFERENCES relationships(relationship_id, investigation_id),
    FOREIGN KEY (paper_id, investigation_id) REFERENCES papers(paper_id, investigation_id),
    FOREIGN KEY (section_id, paper_id, investigation_id) REFERENCES paper_sections(section_id, paper_id, investigation_id)
);

CREATE TABLE IF NOT EXISTS research_gaps (
    gap_id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    gap_description TEXT NOT NULL,
    gap_category TEXT NOT NULL,
    ranking_score REAL NOT NULL,
    ranking_criteria TEXT NOT NULL,
    literature_coverage TEXT,
    limitations TEXT NOT NULL,
    supporting_reasoning TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (gap_id, investigation_id),
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id)
);

CREATE TABLE IF NOT EXISTS gap_supporting_evidence (
    gap_id INTEGER NOT NULL,
    evidence_id INTEGER NOT NULL,
    investigation_id INTEGER NOT NULL,
    PRIMARY KEY (gap_id, evidence_id),
    FOREIGN KEY (gap_id, investigation_id) REFERENCES research_gaps(gap_id, investigation_id),
    FOREIGN KEY (evidence_id, investigation_id) REFERENCES evidence(evidence_id, investigation_id)
);

CREATE TABLE IF NOT EXISTS processing_history (
    task_id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    agent_name TEXT NOT NULL,
    task_key TEXT NOT NULL,
    status TEXT NOT NULL,
    inputs TEXT,
    outputs TEXT,
    checkpoint_state TEXT,
    started_at TEXT,
    completed_at TEXT,
    execution_time_seconds REAL,
    error_details TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 3,
    UNIQUE (task_id, investigation_id),
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_processing_history_task_key ON processing_history(task_key, investigation_id);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
);
INSERT OR IGNORE INTO schema_version VALUES (1, datetime('now'));
