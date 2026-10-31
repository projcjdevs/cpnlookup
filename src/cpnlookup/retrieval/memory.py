"""Conversation turns in ~/.cpnlookup/memory.db (global, so they survive lookup drop)."""
import sqlite3
from contextlib import closing
from datetime import datetime
from cpnlookup.config import get_global_dir

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_global_dir() / "memory.db")
    conn.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, repo TEXT NOT NULL, "
                 "question TEXT NOT NULL, answer TEXT NOT NULL, timestamp TEXT NOT NULL)")
    return conn

def save_turn(repo: str, question: str, answer: str) -> None:
    with closing(_connect()) as conn:
        conn.execute("INSERT INTO conversations (repo, question, answer, timestamp) VALUES (?, ?, ?, ?)", (repo, question, answer, datetime.now().isoformat())); conn.commit()

def get_recent_turns(repo: str, n: int = 4) -> list:
    """The n most recent (question, answer) pairs for repo, oldest first."""
    with closing(_connect()) as conn: rows = conn.execute("SELECT question, answer FROM conversations WHERE repo = ? ORDER BY id DESC LIMIT ?", (repo, n)).fetchall()
    return rows[::-1]

def get_all_history(repo: str) -> list:
    with closing(_connect()) as conn: rows = conn.execute("SELECT id, question, answer, timestamp FROM conversations WHERE repo = ? ORDER BY id DESC", (repo,)).fetchall()
    return [{"id": r[0], "question": r[1], "answer": r[2], "timestamp": r[3]} for r in rows]

def clear_memory(repo: str = None) -> int:
    """Deletes turns for repo (or every repo when None); returns the number deleted."""
    with closing(_connect()) as conn:
        cur = conn.execute("DELETE FROM conversations WHERE repo = ?", (repo,)) if repo else conn.execute("DELETE FROM conversations")
        conn.commit(); return cur.rowcount
