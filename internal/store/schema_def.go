package store

// SchemaSQL defines the database structure
const SchemaSQL = `
-- ========================================================
-- 1. SYSTEM & CONFIG
-- ========================================================
CREATE TABLE IF NOT EXISTS config (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- ========================================================
-- 2. DOCUMENTS & TAGS
-- ========================================================

-- Documents: Metadata shell. Content is stored in 'chunks' or on disk.
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,               -- Display Name
    path TEXT,                        -- Path to cached file
    checksum TEXT,                    -- SHA256 for deduplication
    size INTEGER,                     -- File size in bytes
    type TEXT,                        -- MIME type (application/pdf)
    status TEXT DEFAULT 'pending',    -- 'pending', 'processing', 'indexed', 'failed'
    error_msg TEXT,                   -- If failed, why?
    metadata TEXT,                    -- JSON blob for arbitrary user fields
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tags: User-defined categories.
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE                  -- e.g. "finance"
);

-- Junction: Document <-> Tags
CREATE TABLE IF NOT EXISTS document_tags (
    doc_id INTEGER,
    tag_id INTEGER,
    PRIMARY KEY (doc_id, tag_id),
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE,
    FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
);

-- ========================================================
-- 3. SEARCH & CONTENT
-- ========================================================

-- Chunks: The actual text atoms used for RAG
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER,
    chunk_index INTEGER,              -- 0, 1, 2... order in doc
    start_char_idx INTEGER,           -- Start index in original text
    end_char_idx INTEGER,             -- End index in original text
    content TEXT,                     -- The plain text of this chunk
    embedding BLOB,                   -- Vector (float32 array bytes)
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- FTS5: Keyword Search Index (External Content)
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    content='chunks',
    content_rowid='id'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
  INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ========================================================
-- 4. KNOWLEDGE GRAPH (NER)
-- ========================================================

-- Labels: High-level categories (e.g., "TECHNOLOGY", "PERSON")
CREATE TABLE IF NOT EXISTS labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    global_frequency INTEGER DEFAULT 1
);

-- Entities: Specific instances (e.g., "Java", "Elon Musk")
CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    label_id INTEGER,
    global_frequency INTEGER DEFAULT 1,
    UNIQUE(name, label_id),
    FOREIGN KEY(label_id) REFERENCES labels(id) ON DELETE CASCADE
);

-- Chunk Entities: "Micro-Heatmap" with Offsets
CREATE TABLE IF NOT EXISTS chunk_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER,
    entity_id INTEGER,
    local_frequency INTEGER DEFAULT 1,
    offsets TEXT,                     -- JSON: [[start, end], [start, end]] relative to chunk
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY(entity_id) REFERENCES entities(id) ON DELETE CASCADE
);

-- Chunk Labels: "Macro-Heatmap"
CREATE TABLE IF NOT EXISTS chunk_labels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id INTEGER,
    label_id INTEGER,
    local_frequency INTEGER DEFAULT 1,
    FOREIGN KEY(chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY(label_id) REFERENCES labels(id) ON DELETE CASCADE
);

-- ========================================================
-- 5. LOGGING & AUDIT
-- ========================================================

-- Request Logs: For debugging, optimization, and history
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT,                       -- 'semantic', 'keyword', 'kg_query'
    query_raw TEXT,                  -- The actual text user typed
    query_params TEXT,               -- JSON of filters/settings used
    result_count INTEGER,            -- How many items returned
    latency_ms INTEGER,              -- How long it took
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ========================================================
-- 6. INDEXES
-- ========================================================
CREATE INDEX IF NOT EXISTS idx_documents_checksum ON documents(checksum);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunk_entities_entity ON chunk_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_chunk_entities_chunk ON chunk_entities(chunk_id);
CREATE INDEX IF NOT EXISTS idx_chunk_labels_label ON chunk_labels(label_id);
CREATE INDEX IF NOT EXISTS idx_chunk_labels_chunk ON chunk_labels(chunk_id);
CREATE INDEX IF NOT EXISTS idx_request_logs_type ON request_logs(type);
CREATE INDEX IF NOT EXISTS idx_request_logs_date ON request_logs(created_at);
`
