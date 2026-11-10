-- ============================================================================
-- Migration V002: Literature Acquisition
-- Extends papers table and adds download/import tracking
-- Does NOT modify V001
-- ============================================================================

-- Extend papers table with download tracking fields
ALTER TABLE papers ADD COLUMN normalized_title TEXT;
ALTER TABLE papers ADD COLUMN download_url TEXT;
ALTER TABLE papers ADD COLUMN download_attempted INTEGER NOT NULL DEFAULT 0;
ALTER TABLE papers ADD COLUMN download_error TEXT;
ALTER TABLE papers ADD COLUMN file_size INTEGER;
ALTER TABLE papers ADD COLUMN mime_type TEXT;

-- Index for normalized_title deduplication
CREATE INDEX IF NOT EXISTS idx_papers_normalized_title ON papers(normalized_title, investigation_id) WHERE normalized_title IS NOT NULL;

-- Index for download status queries
CREATE INDEX IF NOT EXISTS idx_papers_access_status ON papers(access_status);
CREATE INDEX IF NOT EXISTS idx_papers_download_attempted ON papers(download_attempted);

-- Paper import tracking (user-uploaded PDFs/TXTs)
CREATE TABLE IF NOT EXISTS paper_imports (
    import_id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    stored_filename TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    mime_type TEXT NOT NULL,
    import_status TEXT NOT NULL DEFAULT 'pending',  -- pending/processed/failed
    import_error TEXT,
    imported_at TEXT NOT NULL,
    paper_id INTEGER,  -- populated once a paper record is created from this import
    UNIQUE (import_id, investigation_id),
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_imports_investigation ON paper_imports(investigation_id);
CREATE INDEX IF NOT EXISTS idx_paper_imports_paper ON paper_imports(paper_id);

-- Paper download attempt log (audit trail)
CREATE TABLE IF NOT EXISTS paper_download_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    investigation_id INTEGER NOT NULL,
    paper_id INTEGER,
    url TEXT NOT NULL,
    attempt_at TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    status_code INTEGER,
    error_message TEXT,
    file_size INTEGER,
    UNIQUE (log_id, investigation_id),
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id),
    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
);

CREATE INDEX IF NOT EXISTS idx_paper_download_logs_paper ON paper_download_logs(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_download_logs_investigation ON paper_download_logs(investigation_id);

-- Update schema version
INSERT OR IGNORE INTO schema_version VALUES (2, datetime('now'));
