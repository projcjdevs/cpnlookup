import sqlite3
import shutil
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
    
    conn.commit()
    conn.close()

def clear_local_index() -> bool:
    local_dir = Path.cwd() / ".cpnlookup"
    if local_dir.exists():
        shutil.rmtree(local_dir)
        return True
    return False