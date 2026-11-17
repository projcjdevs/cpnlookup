"""Hybrid retrieval: FAISS (dense) + SQLite FTS5 BM25 (sparse) -> merge by chunk id -> cross-encoder rerank."""
import re
from cpnlookup.indexer import storage

STOPWORDS = {"the", "an", "is", "are", "was", "how", "does", "do", "did", "what", "where", "which", "who", "why", "when", "in", "of",
             "to", "and", "or", "for", "on", "it", "its", "this", "that", "with", "be", "by", "as", "at", "from", "can", "you", "me", "my"}

def dense_search(query: str, top_k: int) -> list:
    """faiss_ids of the nearest chunks."""
    from cpnlookup.indexer.embedder import embed_texts
    index = storage.load_faiss_index()
    if index is None or index.ntotal == 0: return []
    _, ids = index.search(embed_texts([query]), min(top_k, index.ntotal))
    return [int(i) for i in ids[0] if i >= 0]

def fts_query(text: str) -> str:
    """Word-character terms OR-ed as quoted phrases, so 'fetch_repo_tree' and 'repo tree' both match FTS5 tokens."""
    terms = dict.fromkeys(t for t in re.findall(r"\w+", text.lower()) if len(t) > 1 and t not in STOPWORDS)
    return " OR ".join(f'"{t}"' for t in terms)

def sparse_search(conn, query: str, top_k: int) -> list:
    """Chunk ids ranked by FTS5 bm25(); a hit in the chunk name weighs 10x, in the docstring 2x, in the code 1x."""
    q = fts_query(query)
    if not q: return []
    return [r[0] for r in conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts, 10.0, 2.0, 1.0) LIMIT ?", (q, top_k))]

def merge(dense: list, sparse: list) -> list:
    """Union deduplicated by chunk id, ordered by reciprocal rank fusion: chunks both searches found come first."""
    by_id, score = {}, {}
    for ranked in (dense, sparse):
        for rank, c in enumerate(ranked): by_id.setdefault(c["id"], c); score[c["id"]] = score.get(c["id"], 0) + 1 / (60 + rank)
    return [by_id[i] for i in sorted(score, key=lambda i: -score[i])]

def rerank(query: str, candidates: list, top_k: int) -> list:
    """Cross-encoder scores each (query, chunk) pair jointly; returns the top_k best."""
    if len(candidates) <= 1: return candidates[:top_k]
    from cpnlookup.indexer.embedder import rerank_scores
    scores = rerank_scores([(query, f"File: {c['file_path']}\nName: {c['name']}\nCode: {(c['source_code'] or '')[:800]}") for c in candidates])
    return [c for _, c in sorted(zip(scores, candidates), key=lambda x: -x[0])][:top_k]

def retrieve(conn, query: str, top_k: int, use_rerank: bool = True) -> list:
    """The top_k chunks for a question. Without the reranker the fused dense + sparse order is used as is."""
    fids = dense_search(query, top_k * 2); by_fid = storage.chunks_where(conn, "faiss_id", fids)
    ids = sparse_search(conn, query, top_k * 2); by_id = storage.chunks_where(conn, "id", ids)
    candidates = merge([by_fid[f] for f in fids if f in by_fid], [by_id[i] for i in ids if i in by_id])
    return rerank(query, candidates, top_k) if use_rerank else candidates[:top_k]
