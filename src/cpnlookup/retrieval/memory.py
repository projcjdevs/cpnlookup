"""
Conversation memory for cpnlookup v3.0.0-beta (Flux).

Stores Q&A turns in a persistent SQLite database at ~/.cpnlookup/memory.db.
This is separate from the per-repo index.db so it survives lookup drop.

Intentionally simple in this beta — no summarisation, no embedding of history.
Raw turns are injected into the LLM context as recent conversation context.
Summarisation and history embedding are planned for the full V3 release.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from cpnlookup.utils.config import get_global_dir

def get_memory_db_path() -> Path:
    """Returns path to the global memory database at ~/.cpnlookup/memory.db"""
    db_dir = get_global_dir()
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "memory.db"

def init_memory_db() -> None:
    """Creates the conversations table if it doesn't exist. Safe to call multiple times."""
    conn = sqlite3.connect(get_memory_db_path())
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            repo      TEXT NOT NULL,
            question  TEXT NOT NULL,
            answer    TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit(); conn.close()

def save_turn(repo: str, question: str, answer: str) -> None:
    """Saves a single Q&A turn to memory."""
    init_memory_db()
    conn = sqlite3.connect(get_memory_db_path())
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (repo, question, answer, timestamp) VALUES (?, ?, ?, ?)",
        (repo, question, answer, datetime.now().isoformat())
    )
    conn.commit(); conn.close()

def get_recent_turns(repo: str, n: int = 4) -> list:
    """
    Returns the n most recent Q&A turns for a given repo, oldest first.
    Used to inject prior conversation context into the LLM prompt.

    n=4 by default — enough context without flooding the context window.
    Keeping this small intentionally for the beta; summarisation will
    let us raise this limit safely in the full V3 release.
    """
    try:
        conn = sqlite3.connect(get_memory_db_path())
        cursor = conn.cursor()
        cursor.execute("""
            SELECT question, answer FROM conversations
            WHERE repo = ?
            ORDER BY id DESC LIMIT ?
        """, (repo, n))
        rows = cursor.fetchall()
        conn.close()
        # Reverse so oldest comes first — natural conversation order for the LLM.
        return list(reversed(rows))
    except Exception:
        return []

def get_all_history(repo: str) -> list:
    """
    Returns all stored turns for a repo as [{id, question, answer, timestamp}].
    Used by the lookup history command.
    """
    try:
        init_memory_db()
        conn = sqlite3.connect(get_memory_db_path())
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, question, answer, timestamp FROM conversations
            WHERE repo = ?
            ORDER BY id DESC
        """, (repo,))
        rows = cursor.fetchall()
        conn.close()
        return [{"id": r[0], "question": r[1], "answer": r[2], "timestamp": r[3]} for r in rows]
    except Exception:
        return []

def clear_memory(repo: str) -> int:
    """
    Deletes all conversation history for a given repo.
    Returns the number of rows deleted.
    """
    try:
        conn = sqlite3.connect(get_memory_db_path())
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE repo = ?", (repo,))
        deleted = cursor.rowcount
        conn.commit(); conn.close()
        return deleted
    except Exception:
        return 0

def clear_all_memory() -> int:
    """Wipes the entire memory database. Returns total rows deleted."""
    try:
        conn = sqlite3.connect(get_memory_db_path())
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations")
        deleted = cursor.rowcount
        conn.commit(); conn.close()
        return deleted
    except Exception:
        return 0