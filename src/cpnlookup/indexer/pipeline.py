import os
import re
from contextlib import closing
from cpnlookup import github
from cpnlookup.config import update_registry
from cpnlookup.indexer import chunker, embedder, storage

CALL = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
EMBED_COLS = ("name", "file_path", "chunk_type", "docstring", "source_code")

def build_edges(conn) -> int:
    """Rebuilds every 'calls' edge: identifiers followed by '(' that name another indexed function or class."""
    rows = conn.execute("SELECT id, name, source_code FROM chunks WHERE chunk_type != 'documentation'").fetchall()
    names = {r[1] for r in rows}
    edges = [(cid, t, "calls") for cid, name, src in rows for t in set(CALL.findall(src or "")) & names if t != name]
    conn.execute("DELETE FROM graph_edges"); conn.executemany("INSERT INTO graph_edges (source_id, target_name, edge_type) VALUES (?, ?, ?)", edges)
    return len(edges)

def embed_missing(conn, log) -> int:
    import numpy as np
    todo = conn.execute(f"SELECT id, {', '.join(EMBED_COLS)} FROM chunks WHERE embedding IS NULL").fetchall()
    if not todo: return 0
    log(f"Embedding {len(todo)} chunks (first run downloads the model)...")
    vecs = embedder.embed_chunks([dict(zip(EMBED_COLS, r[1:])) for r in todo])
    conn.executemany("UPDATE chunks SET embedding = ? WHERE id = ?", [(v.astype(np.float32).tobytes(), r[0]) for v, r in zip(vecs, todo)])
    return len(todo)

def rebuild_faiss(conn) -> None:
    """Writes faiss.index from every stored embedding and records each chunk's position in chunks.faiss_id."""
    import numpy as np
    rows = conn.execute("SELECT id, embedding FROM chunks ORDER BY id").fetchall()
    if rows: storage.save_faiss_index(np.vstack([np.frombuffer(e, dtype=np.float32) for _, e in rows]))
    else: storage.faiss_path().unlink(missing_ok=True)
    conn.executemany("UPDATE chunks SET faiss_id = ? WHERE id = ?", [(pos, r[0]) for pos, r in enumerate(rows)])

def run_init(repo: str, log=lambda msg: None) -> dict:
    """Fetches, chunks, graphs and embeds `repo` into ./.cpnlookup incrementally. Raises github.GitHubError."""
    s = github.session(); log("Reading repository tree...")
    tree, truncated = github.fetch_repo_tree(s, repo)  # validates the repo before the index is touched
    upgraded = storage.prepare_index()
    with closing(storage.connect()) as conn:
        old = dict(conn.execute("SELECT file_path, file_hash FROM raw_files").fetchall())
        paths = {f["path"] for f in tree}
        changed, removed = [f for f in tree if old.get(f["path"]) != f["sha"]], [p for p in old if p not in paths]
        result = {"repo": repo, "fresh": not old, "changed": len(changed), "removed": len(removed), "truncated": truncated, "upgraded": upgraded, "up_to_date": False}
        if not changed and not removed and storage.get_meta(conn, "index_status") == "complete" and storage.get_meta(conn, "repo") == repo:
            result["up_to_date"] = True; return result
        storage.set_meta(conn, "index_status", "pending"); storage.set_meta(conn, "repo", repo); conn.commit()
        for p in removed: storage.delete_file_data(conn, p)
        for i, f in enumerate(changed, 1):
            log(f"Fetching {i}/{len(changed)}: {f['path']}")
            content = github.fetch_blob(s, repo, f["sha"])
            storage.delete_file_data(conn, f["path"])
            conn.execute("INSERT INTO raw_files (file_path, file_hash) VALUES (?, ?)", (f["path"], f["sha"]))
            for c in chunker.chunk_file(f["path"], content or ""):
                conn.execute("INSERT INTO chunks (name, file_path, line_start, line_end, chunk_type, source_code, docstring) VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (c["name"], c["file_path"], c["line_start"], c["line_end"], c["chunk_type"], c["source_code"], c["docstring"]))
            conn.commit()
        log("Building call graph..."); result["edges"] = build_edges(conn)
        result["embedded"] = embed_missing(conn, log); conn.commit()
        log("Writing vector index..."); rebuild_faiss(conn)
        result["total"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        storage.set_meta(conn, "index_status", "complete"); conn.commit()
    update_registry(os.getcwd(), repo)
    return result
