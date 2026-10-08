CREATE TABLE IF NOT EXISTS documents (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    format       TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    loaded_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    content     TEXT NOT NULL,
    char_count  INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(id),
    UNIQUE(document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS query_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    asked_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    question         TEXT NOT NULL,
    answer           TEXT,
    source_chunk_ids TEXT,
    best_distance    REAL
);