from click.testing import CliRunner
from cpnlookup import __version__
from cpnlookup.cli import cli

def run(*args, **kw): return CliRunner().invoke(cli, list(args), **kw)

def test_version_and_help():
    assert run("--version").output.strip() == f"cpnlookup, version {__version__}"
    out = run("help").output
    assert "lookup init <owner/repo>" in out and "lookup forget [--all]" in out
    assert run("commands").output == out
    assert "Flux" in run("desc").output

def test_every_command_has_help_text():
    listing = run("--help").output
    for name in ("auth", "profile", "init", "ask", "functions", "indexed", "config", "clone", "drop", "history", "forget", "desc", "help"):
        line = next(l for l in listing.splitlines() if l.strip().startswith(name + " "))
        assert len(line.split(None, 1)[1]) > 5, name

def test_errors_exit_nonzero_without_tracebacks(isolated):
    for args in (["ask", "x"], ["functions"], ["config", "top_k", "5"], ["history"], ["drop"], ["init", "not-a-repo"]):
        r = run(*args); assert r.exit_code == 1 and "Error:" in r.output and "Traceback" not in r.output, (args, r.output)
    assert not (isolated / ".cpnlookup").exists()                                # no stray index dirs

def test_config_show_and_set(isolated):
    (isolated / ".cpnlookup").mkdir()
    assert run("config", "top_k", "9").exit_code == 0
    r = run("config", "top_k", "lots"); assert r.exit_code == 1 and "whole number" in r.output
    assert "9" in run("config").output

def test_ask_prints_answer_literally_and_never_saves_errors(gh, monkeypatch):
    from cpnlookup import llm
    from cpnlookup.retrieval import search, memory
    assert run("init", "o/r").exit_code == 0
    monkeypatch.setattr(llm, "check_ollama", lambda: True)
    monkeypatch.setattr(search, "dense_search", lambda query, k: [])
    monkeypatch.setattr(search, "rerank", lambda query, cands, k: cands[:k])
    monkeypatch.setattr(llm, "chat_with_ollama", lambda prompt, **kw: "use items[key] and x[i], then [/b]")
    r = run("ask", "what is [bold]parse?")
    assert r.exit_code == 0 and "use items[key] and x[i], then [/b]" in r.output and "[bold]parse?" in r.output
    def boom(prompt, **kw): raise llm.OllamaError("Ollama returned 404: model 'nope' not found. Try: ollama pull nope")
    monkeypatch.setattr(llm, "chat_with_ollama", boom)
    r = run("ask", "second?"); assert r.exit_code == 1 and "ollama pull nope" in r.output
    assert [t["question"] for t in memory.get_all_history("o/r")] == ["what is [bold]parse?"]
    assert "History" in run("history").output and "Cleared 1" in run("forget", input="y\n").output

def test_profile_all_flag(monkeypatch):
    from cpnlookup import github
    seen = []
    def fake(user, limit): seen.append(limit); return [{"name": "r[1]", "stargazers_count": 3, "language": None}]
    monkeypatch.setattr(github, "get_user_repos", fake)
    assert "r[1]" in run("profile", "someone", "--all").output and "r[1]" in run("profile", "someone").output
    assert seen == [None, 20]
