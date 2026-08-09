"""
Hybrid retrieval pipeline for cpnlookup v3.0.0-beta (Flux).

Adds two retrieval layers on top of V2's FAISS search:
  1. BM25 sparse search  — exact keyword / symbol name matching
  2. Cross-encoder reranking — joint query+chunk scoring for precision

Flow:
  FAISS (dense)  ──┐
                   ├──> merge candidates ──> cross-encoder rerank ──> top-k
  BM25  (sparse) ──┘
"""

def _build_bm25(chunks: list):
    """
    Builds a BM25 index over a list of chunk dicts.
    Each chunk is tokenised from its name + source_code + docstring.
    Lazy-imported so rank_bm25 only loads when retrieval actually runs.
    """
    from rank_bm25 import BM25Okapi
    corpus = []
    for c in chunks:
        text = f"{c.get('name', '')} {c.get('docstring', '')} {c.get('source_code', '')}"
        tokens = text.lower().split()
        corpus.append(tokens)
    return BM25Okapi(corpus)

def bm25_search(query: str, chunks: list, top_k: int = 10) -> list:
    """
    Runs BM25 keyword search over chunks.
    Returns a list of (chunk_index, score) tuples sorted descending.

    Particularly strong on exact symbol name queries like
    'fetch_repo_tree' where dense search may miss the exact match.
    """
    import numpy as np
    bm25 = _build_bm25(chunks)
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

def rerank(query: str, candidates: list, top_k: int = 5) -> list:
    """
    Reranks a list of candidate chunk dicts using a cross-encoder.

    A cross-encoder scores (query, chunk) jointly — it reads both at once
    rather than comparing independent embeddings. This catches cases where
    a chunk is superficially similar to the query but actually irrelevant,
    or vice versa.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2 (~85MB, downloaded once).
    Returns the top_k candidates sorted by cross-encoder score descending.
    """
    if not candidates:
        return []

    from sentence_transformers import CrossEncoder
    from cpnlookup.utils.config import get_global_dir

    model_cache = str(get_global_dir() / "models")
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)

    # Build (query, chunk_text) pairs for the cross-encoder.
    pairs = []
    for c in candidates:
        chunk_text = (
            f"File: {c.get('file_path', '')}\n"
            f"Name: {c.get('name', '')}\n"
            f"Code: {c.get('source_code', '')[:800]}"  # cap at 800 chars to stay within max_length
        )
        pairs.append((query, chunk_text))

    scores = model.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in ranked[:top_k]]

def merge_results(faiss_ids: list, bm25_results: list, all_chunks: list) -> list:
    """
    Merges FAISS and BM25 results using a simple union with deduplication.

    faiss_ids: list of faiss_id integers from vector_search
    bm25_results: list of (chunk_list_index, score) from bm25_search
    all_chunks: the full ordered list of chunk dicts (index matches faiss_id)

    Returns a deduplicated list of chunk dicts for reranking.
    Uses a seen set on chunk name to avoid feeding duplicate context to
    the cross-encoder.
    """
    seen, merged = set(), []

    # FAISS results first — they carry semantic relevance.
    for faiss_id in faiss_ids:
        if 0 <= faiss_id < len(all_chunks):
            c = all_chunks[faiss_id]
            key = (c.get('name'), c.get('file_path'))
            if key not in seen:
                merged.append(c)
                seen.add(key)

    # BM25 results second — they add keyword-matched chunks FAISS may have missed.
    for idx, score in bm25_results:
        if 0 <= idx < len(all_chunks):
            c = all_chunks[idx]
            key = (c.get('name'), c.get('file_path'))
            if key not in seen:
                merged.append(c)
                seen.add(key)

    return merged