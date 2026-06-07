import sqlite3
import shutil
import os
from pathlib import Path

def get_local_db_path() -> Path:
    local_dir = Path.cwd() / ".cpnlookup"
    local_dir.mkdir(parents=True, exist_ok=True)
    return local_dir / "index.db"

def init_db() -> None:
    db_path = get_local_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS raw_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL UNIQUE,
            language TEXT,
            content TEXT,
            size_bytes INTEGER,
            file_hash TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            line_start INTEGER,
            line_end INTEGER,
            chunk_type TEXT,
            source_code TEXT,
            docstring TEXT,
            embedding BLOB,
            faiss_id INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_id INTEGER REFERENCES chunks(id),
            name TEXT NOT NULL,
            file_path TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS graph_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER REFERENCES graph_nodes(id),
            target_name TEXT NOT NULL,
            edge_type TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS repo_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    for migration in [
        "ALTER TABLE chunks ADD COLUMN faiss_id INTEGER",
        "ALTER TABLE chunks ADD COLUMN embedding BLOB",
        "ALTER TABLE raw_files ADD COLUMN file_hash TEXT",
    ]:
        try:
            cursor.execute(migration)
        except sqlite3.OperationalError:
            pass

    conn.commit()
    conn.close()

def set_index_status(status: str) -> None:
    db_path = get_local_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO repo_meta (key, value) VALUES ('index_status', ?)", (status,))
    conn.commit(); conn.close()

def get_index_status() -> str:
    db_path = get_local_db_path()
    if not db_path.exists(): return None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT value FROM repo_meta WHERE key = 'index_status'")
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        conn.close(); return None

def load_file_hashes() -> dict:
    db_path = get_local_db_path()
    if not db_path.exists(): return {}
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT file_path, file_hash FROM raw_files")
        result = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close(); return result
    except Exception:
        conn.close(); return {}

def delete_file_data(cursor, file_path: str) -> None:
    cursor.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,))
    chunk_ids = [r[0] for r in cursor.fetchall()]
    if chunk_ids:
        ph = ','.join('?' * len(chunk_ids))
        cursor.execute(f"SELECT id FROM graph_nodes WHERE chunk_id IN ({ph})", chunk_ids)
        node_ids = [r[0] for r in cursor.fetchall()]
        if node_ids:
            ph2 = ','.join('?' * len(node_ids))
            cursor.execute(f"DELETE FROM graph_edges WHERE source_id IN ({ph2})", node_ids)
        cursor.execute(f"DELETE FROM graph_nodes WHERE chunk_id IN ({ph})", chunk_ids)
    cursor.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
    cursor.execute("DELETE FROM raw_files WHERE file_path = ?", (file_path,))

def save_faiss_index(embeddings) -> None:
    import faiss
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    faiss.write_index(index, str(Path.cwd() / ".cpnlookup" / "faiss.index"))

def load_faiss_index():
    import faiss
    index_path = Path.cwd() / ".cpnlookup" / "faiss.index"
    if not index_path.exists(): return None
    return faiss.read_index(str(index_path))

def clear_local_index() -> bool:
    local_dir = Path.cwd() / ".cpnlookup"
    if local_dir.exists():
        shutil.rmtree(local_dir); return True
    return False