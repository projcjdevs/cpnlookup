"""Retrieval quality on a golden question set. Run inside a directory indexed with `lookup init <repo>`:

    python eval/retrieval_eval.py eval/golden_cpnlookup.json [--json]

Reports hit rate and MRR at top_k for dense search alone, sparse search alone, both fused without the reranker
(`rerank off`) and the full retrieve() pipeline.
"""
import json
import sys
from contextlib import closing
from pathlib import Path
from cpnlookup.indexer import storage
from cpnlookup.retrieval import search

TOP_K = 5

def rank_of(chunks: list, expect: set) -> int:
    """1-based rank of the first expected chunk, or 0 when none was retrieved."""
    return next((i for i, c in enumerate(chunks, 1) if (c["file_path"], c["name"]) in expect), 0)

def stages(conn, q: str) -> dict:
    fids = search.dense_search(q, TOP_K); by_fid = storage.chunks_where(conn, "faiss_id", fids)
    ids = search.sparse_search(conn, q, TOP_K); by_id = storage.chunks_where(conn, "id", ids)
    return {"dense": [by_fid[f] for f in fids if f in by_fid], "sparse": [by_id[i] for i in ids if i in by_id],
            "no_rerank": search.retrieve(conn, q, TOP_K, use_rerank=False), "retrieve": search.retrieve(conn, q, TOP_K)}

def evaluate(golden: dict) -> dict:
    """{stage: {hit_rate, mrr, ranks}} over every golden question."""
    ranks = {"dense": [], "sparse": [], "no_rerank": [], "retrieve": []}
    with closing(storage.connect()) as conn:
        for item in golden["questions"]:
            expect = {tuple(e) for e in item["expect"]}
            for stage, chunks in stages(conn, item["q"]).items(): ranks[stage].append(rank_of(chunks, expect))
    n = len(golden["questions"])
    return {s: {"hit_rate": sum(r > 0 for r in rs) / n, "mrr": sum(1 / r for r in rs if r) / n, "ranks": rs} for s, rs in ranks.items()}

def main(argv: list) -> int:
    if not argv: print(__doc__); return 2
    problem = storage.index_problem()
    if problem: print(problem); return 1
    golden = json.loads(Path(argv[0]).read_text(encoding="utf-8")); result = evaluate(golden)
    if "--json" in argv: print(json.dumps(result)); return 0
    print(f"{len(golden['questions'])} questions on {golden['repo']}, top_k={TOP_K}")
    for stage, r in result.items(): print(f"{stage:9} hit@{TOP_K} {r['hit_rate']:.2f}   MRR {r['mrr']:.3f}   ranks {r['ranks']}")
    for item, rank in zip(golden["questions"], result["retrieve"]["ranks"]):
        if not rank: print(f"  MISS: {item['q']}")
    return 0

if __name__ == "__main__": sys.exit(main(sys.argv[1:]))
