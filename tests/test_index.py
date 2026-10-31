import sqlite3
from contextlib import closing
import pytest
from cpnlookup.github import GitHubError
from cpnlookup.indexer import storage
from cpnlookup.indexer.pipeline import run_init
from cpnlookup.retrieval import search
from cpnlookup.retrieval.context import expand, build_prompt

def q(sql, *args):
    with closing(storage.connect()) as conn: return conn.execute(sql, args).fetchall()

def test_fresh_index(gh):
    r = run_init("o/r")
    assert r["fresh"] and not r["up_to_date"] and r["total"] == q("SELECT COUNT(*) FROM chunks")[0][0] > 0
    assert storage.index_problem() is None and storage.current_repo() == "o/r"
    fids = [row[0] for row in q("SELECT faiss_id FROM chunks ORDER BY faiss_id")]
    assert fids == list(range(len(fids))) and storage.load_faiss_index().ntotal == len(fids)
    assert q("SELECT COUNT(*) FROM raw_files")[0][0] == 4
    assert {row[0] for row in q("SELECT name FROM sqlite_master")}.isdisjoint({"graph_nodes"})

def test_edges_use_identifier_tokens(gh):
    run_init("o/r")
    edges = set(q("SELECT c.name, e.target_name FROM graph_edges e JOIN chunks c ON c.id = e.source_id"))
    assert ("fetch_repo_tree", "parse") in edges and ("__init__", "fetch_repo_tree") in edges and ("run", "parse") in edges
    assert not any(t == "parse" and s == "my_parse" for s, t in edges)        # no substring match of my_parse(
    assert not any(s == t for s, t in edges)

def test_rerun_is_incremental(gh):
    run_init("o/r"); first_fetch, first_embed = list(gh.fetched), sum(gh.embed_calls)
    assert run_init("o/r")["up_to_date"]
    gh.files["pkg/other.py"] = "def brand_new():\n    return fetch_repo_tree('x')\n"; del gh.files["README.md"]
    r = run_init("o/r")
    assert (r["changed"], r["removed"]) == (1, 1) and gh.fetched[len(first_fetch):] == ["pkg/other.py"]
    assert sum(gh.embed_calls) - first_embed == 1                               # only brand_new embedded
    assert not q("SELECT 1 FROM chunks WHERE file_path = 'README.md'") and not q("SELECT 1 FROM chunks_fts WHERE chunks_fts MATCH 'install'")
    assert ("brand_new", "fetch_repo_tree") in set(q("SELECT c.name, e.target_name FROM graph_edges e JOIN chunks c ON c.id = e.source_id"))
    fids = [row[0] for row in q("SELECT faiss_id FROM chunks ORDER BY faiss_id")]
    assert fids == list(range(len(fids))) and storage.load_faiss_index().ntotal == len(fids)

def test_github_failure_leaves_index_pending_without_empty_files(gh):
    gh.fail_on = "pkg/other.py"
    with pytest.raises(GitHubError): run_init("o/r")
    assert "incomplete" in storage.index_problem()
    assert not q("SELECT 1 FROM raw_files WHERE file_path = 'pkg/other.py'")
    gh.fail_on = None; run_init("o/r"); assert storage.index_problem() is None

def test_bad_repo_does_not_touch_existing_index(gh, monkeypatch):
    run_init("o/r")
    def missing(s, repo): raise GitHubError("Not found")
    monkeypatch.setattr("cpnlookup.github.fetch_repo_tree", missing)
    with pytest.raises(GitHubError): run_init("o/typo")
    assert storage.index_problem() is None and storage.current_repo() == "o/r"

def test_old_schema_is_rebuilt(gh, isolated):
    (isolated / ".cpnlookup").mkdir()
    with closing(sqlite3.connect(storage.db_path())) as conn:
        conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, name TEXT)"); conn.execute("CREATE TABLE graph_nodes (id INTEGER)"); conn.commit()
    assert "older cpnlookup" in storage.index_problem()
    r = run_init("o/r")
    assert r["upgraded"] and r["fresh"] and storage.index_problem() is None

def test_sparse_search_matches_identifiers_and_words(gh):
    run_init("o/r")
    with closing(storage.connect()) as conn:
        for query in ("fetch_repo_tree", "where is the repo tree fetched"):
            top = storage.chunks_where(conn, "id", search.sparse_search(conn, query, 3))
            assert "fetch_repo_tree" in {c["name"] for c in top.values()}

def test_retrieve_maps_faiss_ids_through_column(gh, monkeypatch):
    run_init("o/r")
    target = q("SELECT faiss_id, name FROM chunks WHERE name = 'my_parse'")[0]
    monkeypatch.setattr(search, "dense_search", lambda query, k: [target[0], 999])
    monkeypatch.setattr(search, "rerank", lambda query, cands, k: cands[:k])
    with closing(storage.connect()) as conn: got = search.retrieve(conn, "zzz_nothing", 5)
    assert [c["name"] for c in got] == ["my_parse"]

def test_expand_caps_neighbors_and_keeps_same_named_chunks(gh):
    run_init("o/r")
    with closing(storage.connect()) as conn:
        inits = storage.rows_to_chunks(conn.execute(f"SELECT {', '.join(storage.CHUNK_COLS)} FROM chunks WHERE name = '__init__'"))
        items = expand(conn, inits, max_neighbors=1)
        parse = storage.rows_to_chunks(conn.execute(f"SELECT {', '.join(storage.CHUNK_COLS)} FROM chunks WHERE name = 'parse'"))
        callers = [c for label, c in expand(conn, parse, max_neighbors=1) if label == "CALLER"]
    assert sum(label == "MATCH" for label, _ in items) == 2                       # both __init__ kept
    assert any(label == "CALLEE" and c["name"] == "fetch_repo_tree" for label, c in items)
    assert len(callers) == 1                                                     # capped

def test_build_prompt_budget_keeps_first_match():
    big = {"file_path": "a.py", "line_start": 1, "name": "big", "source_code": "x" * 50_000}
    small = {"file_path": "b.py", "line_start": 1, "name": "small", "source_code": "y" * 100}
    prompt = build_prompt("Q?", [("MATCH", big), ("CALLER", small)], [("old q", "a" * 5000)], num_ctx=2048)
    assert "--- MATCH: a.py:1 | big ---" in prompt and "small" not in prompt
    assert len(prompt) < 2048 * 2.5 + 2000 and prompt.endswith("Question: Q?") and "a" * 1001 not in prompt
