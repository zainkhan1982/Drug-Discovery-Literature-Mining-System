-- ============================================================================
-- Migration V003: Biomedical NLP and Relationship Extraction
-- Adds processing_status tracking and indexes for NLP tables
-- Does NOT modify V001 or V002
-- ============================================================================

-- Add processing_status to papers table
ALTER TABLE papers ADD COLUMN processing_status TEXT NOT NULL DEFAULT 'pending';

-- Index for processing status queries
CREATE INDEX IF NOT EXISTS idx_papers_processing_status ON papers(processing_status);
CREATE INDEX IF NOT EXISTS idx_papers_processing_inv ON papers(investigation_id, processing_status);

-- Add external_identifier columns to entities table
ALTER TABLE entities ADD COLUMN external_identifier TEXT;
ALTER TABLE entities ADD COLUMN external_identifier_source TEXT;

-- Indexes for entity lookups
CREATE INDEX IF NOT EXISTS idx_entities_inv_type ON entities(investigation_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_normalized_inv ON entities(normalized_name, investigation_id) WHERE normalized_name IS NOT NULL;

-- Indexes for mention lookups
CREATE INDEX IF NOT EXISTS idx_mentions_inv_entity ON paper_entity_mentions(investigation_id, entity_id);
CREATE INDEX IF NOT EXISTS idx_mentions_inv_paper ON paper_entity_mentions(investigation_id, paper_id);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON paper_entity_mentions(chunk_id);

-- Indexes for relationship lookups
CREATE INDEX IF NOT EXISTS idx_relationships_inv_type ON relationships(investigation_id, relationship_type);
CREATE INDEX IF NOT EXISTS idx_relationships_inv_entities ON relationships(investigation_id, source_entity_id, target_entity_id);

-- Indexes for evidence lookups
CREATE INDEX IF NOT EXISTS idx_evidence_inv_relationship ON evidence(investigation_id, relationship_id);
CREATE INDEX IF NOT EXISTS idx_evidence_inv_paper ON evidence(investigation_id, paper_id);

-- Indexes for chunk lookups
CREATE INDEX IF NOT EXISTS idx_chunks_inv_paper ON paper_chunks(investigation_id, paper_id);

-- Update schema version
INSERT OR IGNORE INTO schema_version VALUES (3, datetime('now'));
