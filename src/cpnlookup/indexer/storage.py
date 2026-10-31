import shutil
import sqlite3
from contextlib import closing
from pathlib import Path
from cpnlookup.config import local_dir, registry_lookup

SCHEMA_VERSION = "3"
SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_files (file_path TEXT PRIMARY KEY, file_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, file_path TEXT NOT NULL,
    line_start INTEGER, line_end INTEGER, chunk_type TEXT, source_code TEXT, docstring TEXT,
    embedding BLOB, faiss_id INTEGER);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_name ON chunks(name);
CREATE INDEX IF NOT EXISTS idx_chunks_faiss ON chunks(faiss_id);
CREATE TABLE IF NOT EXISTS graph_edges (source_id INTEGER NOT NULL REFERENCES chunks(id), target_name TEXT NOT NULL, edge_type TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_edges_source ON graph_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON graph_edges(target_name);
CREATE TABLE IF NOT EXISTS repo_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(name, docstring, source_code, content='chunks', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, name, docstring, source_code) VALUES (new.id, new.name, new.docstring, new.source_code); END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, name, docstring, source_code) VALUES ('delete', old.id, old.name, old.docstring, old.source_code); END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF name, docstring, source_code ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, name, docstring, source_code) VALUES ('delete', old.id, old.name, old.docstring, old.source_code);
    INSERT INTO chunks_fts(rowid, name, docstring, source_code) VALUES (new.id, new.name, new.docstring, new.source_code); END;
"""
CHUNK_COLS = ("id", "faiss_id", "name", "file_path", "line_start", "line_end", "chunk_type", "docstring", "source_code")

def db_path() -> Path: return local_dir() / "index.db"
def faiss_path() -> Path: return local_dir() / "faiss.index"
def has_index() -> bool: return db_path().exists()

def connect() -> sqlite3.Connection: return sqlite3.connect(db_path())

def get_meta(conn, key: str):
    try: row = conn.execute("SELECT value FROM repo_meta WHERE key = ?", (key,)).fetchone()
    except sqlite3.OperationalError: return None
    return row[0] if row else None

def set_meta(conn, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO repo_meta (key, value) VALUES (?, ?)", (key, value))

def prepare_index() -> bool:
    """Creates or opens the v3 index. An index from an older schema is deleted first; returns True if that happened."""
    upgraded = False
    if has_index():
        with closing(connect()) as conn: upgraded = get_meta(conn, "schema_version") != SCHEMA_VERSION
        if upgraded: db_path().unlink(); faiss_path().unlink(missing_ok=True)
    local_dir().mkdir(parents=True, exist_ok=True)
    with closing(connect()) as conn: conn.executescript(SCHEMA); set_meta(conn, "schema_version", SCHEMA_VERSION); conn.commit()
    return upgraded

def index_problem():
    """Returns a user-facing message if this directory has no usable index, else None."""
    if not has_index(): return "No index in this directory. Run [bold]lookup init <owner/repo>[/] first."
    with closing(connect()) as conn: version, status = get_meta(conn, "schema_version"), get_meta(conn, "index_status")
    if version != SCHEMA_VERSION: return "This index was built by an older cpnlookup. Run [bold]lookup init[/] to rebuild it."
    if status != "complete": return "The index is incomplete (an earlier run stopped). Run [bold]lookup init[/] again."
    return None

def current_repo():
    """Repo of this directory's index (repo_meta), falling back to the global registry."""
    if has_index():
        with closing(connect()) as conn: repo = get_meta(conn, "repo")
        if repo: return repo
    return registry_lookup(str(Path.cwd()))

def rows_to_chunks(rows) -> list: return [dict(zip(CHUNK_COLS, r)) for r in rows]

def chunks_where(conn, column: str, values) -> dict:
    """Maps column value -> chunk dict for rows whose `column` is in `values` (column: 'id' or 'faiss_id')."""
    values = list(values)
    if not values: return {}
    rows = conn.execute(f"SELECT {', '.join(CHUNK_COLS)} FROM chunks WHERE {column} IN ({','.join('?' * len(values))})", values)
    return {c[column]: c for c in rows_to_chunks(rows)}

def delete_file_data(conn, file_path: str) -> None:
    conn.execute("DELETE FROM graph_edges WHERE source_id IN (SELECT id FROM chunks WHERE file_path = ?)", (file_path,))
    conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,)); conn.execute("DELETE FROM raw_files WHERE file_path = ?", (file_path,))

def save_faiss_index(embeddings) -> None:
    import faiss
    index = faiss.IndexFlatL2(embeddings.shape[1]); index.add(embeddings.astype("float32")); faiss.write_index(index, str(faiss_path()))

def load_faiss_index():
    import faiss
    return faiss.read_index(str(faiss_path())) if faiss_path().exists() else None

def clear_local_index() -> bool:
    if not local_dir().exists(): return False
    shutil.rmtree(local_dir()); return True
