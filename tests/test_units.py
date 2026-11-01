import pytest
from cpnlookup.config import set_local_config, get_local_config, update_registry, get_registry, registry_lookup
from cpnlookup.github import should_skip_file, parse_repo, GitHubError
from cpnlookup.indexer.chunker import chunk_python_code, chunk_markdown
from cpnlookup.retrieval.search import fts_query, merge

@pytest.mark.parametrize("path,skip", [
    ("src/app.py", False), ("tests/test_app.py", False), ("latest.py", False), ("docs/Guide.MD", False),
    ("venv/lib/site/tests/test_x.py", True), ("node_modules/x/test/a.py", True), ("pkg.egg-info/x.py", True),
    ("build/lib/a.py", True), ("logo.png", True), ("src/main.js", True)])
def test_should_skip_file(path, skip): assert should_skip_file(path) is skip

@pytest.mark.parametrize("text", ["owner/repo", "https://github.com/owner/repo", "github.com/owner/repo/", "https://github.com/owner/repo.git", "git@github.com:owner/repo.git"])
def test_parse_repo_forms(text): assert parse_repo(text) == "owner/repo"

@pytest.mark.parametrize("text", ["repo", "a/b/c", "https://gitlab.com/a/b", ""])
def test_parse_repo_rejects(text):
    with pytest.raises(GitHubError): parse_repo(text)

def test_python_chunks_include_methods_and_async():
    code = 'class A:\n    """Doc."""\n    def m(self): pass\n\nasync def go():\n    return 1\n'
    got = {(c["name"], c["chunk_type"]) for c in chunk_python_code("a.py", code)}
    assert got == {("A", "class"), ("m", "function"), ("go", "function")}
    assert chunk_python_code("bad.py", "def (:") == []

def test_markdown_sections_lines_and_fences():
    md = "# Title\nIntro\n```bash\n# not a header\n```\n## Use\nRun it\n"
    chunks = chunk_markdown("docs/guide.md", md)
    assert [c["name"] for c in chunks] == ["guide: Title", "guide: Use"]
    assert "# not a header" in chunks[0]["source_code"] and (chunks[1]["line_start"], chunks[1]["line_end"]) == (6, 7)

def test_fts_query_sanitizes():
    assert fts_query('Where is fetch_repo_tree() defined?') == '"fetch_repo_tree" OR "defined"'
    assert fts_query("how does the?") == ""

def test_merge_dedups_by_id_not_name():
    a, b, a2 = {"id": 1, "name": "__init__"}, {"id": 2, "name": "__init__"}, {"id": 1, "name": "__init__"}
    assert merge([a], [b, a2]) == [a, b]

def test_config_validation(isolated):
    with pytest.raises(ValueError, match="No index"): set_local_config("model", "llama3")
    (isolated / ".cpnlookup").mkdir()
    with pytest.raises(ValueError, match="Unknown setting"): set_local_config("modle", "x")
    with pytest.raises(ValueError, match="whole number"): set_local_config("top_k", "abc")
    with pytest.raises(ValueError, match="at least 1"): set_local_config("top_k", "0")
    assert set_local_config("top_k", "7") == 7 and get_local_config()["top_k"] == 7 and get_local_config()["num_ctx"] == 8192

def test_registry_merges_equivalent_paths(isolated):
    update_registry(str(isolated), "a/b"); update_registry(str(isolated) + "/.", "c/d")
    assert list(get_registry().values()) == ["c/d"] and registry_lookup(str(isolated)) == "c/d"
    update_registry(str(isolated), "", add=False); assert get_registry() == {}
