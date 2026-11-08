CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    orig_path TEXT NOT NULL,
    work_path TEXT NOT NULL,
    is_sample INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    image_id TEXT NOT NULL REFERENCES images(id),
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_methods (
    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    method_id TEXT NOT NULL,
    status TEXT NOT NULL,
    count INTEGER,
    points_json TEXT,
    grid_boxes_json TEXT,
    rendered_path TEXT,
    elapsed_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    model_size_mb REAL,
    meta_json TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, method_id)
);

CREATE TABLE IF NOT EXISTS model_packs (
    slug TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    aliases_json TEXT NOT NULL,
    onnx_path TEXT NOT NULL,
    size_mb REAL NOT NULL,
    imgsz INTEGER NOT NULL,
    trained_epochs INTEGER NOT NULL,
    val_mape_pct REAL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
