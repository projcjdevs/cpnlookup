import sqlite3
import shutil
import faiss
import numpy as np
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
            size_bytes INTEGER
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
            embedding BLOB
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
    
    conn.commit()
    conn.close()

def clear_local_index() -> bool:
    local_dir = Path.cwd() / ".cpnlookup"
    if local_dir.exists():
        shutil.rmtree(local_dir)
        return True
    return False

def save_faiss_index(embeddings: np.ndarray):
    """Saves the embeddings matrix to .cpnlookup/faiss.index."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    index_path = Path.cwd() / ".cpnlookup" / "faiss.index"
    faiss.write_index(index, str(index_path))

def load_faiss_index():
    """Loads the FAISS index from the local project folder."""
    index_path = Path.cwd() / ".cpnlookup" / "faiss.index"
    if not index_path.exists():
        return None
    return faiss.read_index(str(index_path))