import pytest
from click.testing import CliRunner
from cpnlookup import __version__
from cpnlookup.cli import cli

def run(*args, **kw): return CliRunner().invoke(cli, list(args), **kw)

def test_version_and_help():
    assert run("--version").output.strip() == f"cpnlookup, version {__version__}"
    out = run("help").output
    assert "lookup init <owner/repo>" in out and "lookup forget [--all]" in out
    assert run("commands").output == out
    desc = run("desc").output; assert "Flux" in desc and "Featherweight" in desc

def test_every_command_has_help_text():
    listing = run("--help").output
    for name in ("auth", "profile", "init", "ask", "chat", "functions", "indexed", "config", "clone", "drop", "history", "forget", "desc", "help"):
        line = next(l for l in listing.splitlines() if l.strip().startswith(name + " "))
        assert len(line.split(None, 1)[1]) > 5, name

def test_errors_exit_nonzero_without_tracebacks(isolated):
    for args in (["ask", "x"], ["chat"], ["functions"], ["config", "top_k", "5"], ["history"], ["drop"], ["init", "not-a-repo"]):
        r = run(*args); assert r.exit_code == 1 and "Error:" in r.output and "Traceback" not in r.output, (args, r.output)
    assert not (isolated / ".cpnlookup").exists()                                # no stray index dirs

def test_config_show_and_set(isolated):
    (isolated / ".cpnlookup").mkdir()
    assert run("config", "top_k", "9").exit_code == 0
    r = run("config", "top_k", "lots"); assert r.exit_code == 1 and "whole number" in r.output
    assert "9" in run("config").output

@pytest.fixture
def ready(gh, monkeypatch):
    """An indexed repo with retrieval and Ollama faked; ready.reply is what the model 'writes' (an exception instance is raised mid-answer)."""
    from cpnlookup import llm
    from cpnlookup.retrieval import search
    assert run("init", "o/r").exit_code == 0
    state = type("S", (), {"reply": ["ok"], "prompts": [], "reranked": 0})()
    def stream(prompt, **kw):
        state.prompts.append(prompt)
        for piece in state.reply:
            if isinstance(piece, BaseException): raise piece
            yield piece
    def rerank(query, cands, k): state.reranked += 1; return cands[:k]
    monkeypatch.setattr(llm, "ensure_ready", lambda model: None); monkeypatch.setattr(llm, "stream_ollama", stream)
    monkeypatch.setattr(search, "dense_search", lambda query, k: []); monkeypatch.setattr(search, "rerank", rerank)
    return state

def test_ask_streams_literally_cites_sources_and_never_saves_failures(ready):
    from cpnlookup import llm
    from cpnlookup.retrieval import memory
    ready.reply = ["use items[key] and x[i], ", "then [/b]"]
    r = run("ask", "what does [bold]parse do?")
    assert r.exit_code == 0 and "use items[key] and x[i], then [/b]" in r.output and "[bold]parse do?" in r.output
    assert "Sources:" in r.output and "pkg/fetch.py:5-6" in r.output and "pkg/fetch.py:5 | parse" in ready.prompts[0]
    assert "Took " in r.output and "waiting for the model" in r.output
    assert "related (callers and callees): " in r.output and r.output.count("pkg/fetch.py:5-6") == 1
    ready.reply = ["half an ans", llm.OllamaError("Ollama stopped: out of memory")]
    r = run("ask", "second?"); assert r.exit_code == 1 and "half an ans" in r.output and "out of memory" in r.output and "Sources:" not in r.output
    ready.reply = ["  \n"]
    r = run("ask", "third?"); assert r.exit_code == 1 and "empty answer" in r.output
    assert [t["question"] for t in memory.get_all_history("o/r")] == ["what does [bold]parse do?"]
    assert "History" in run("history").output and "Cleared 1" in run("forget", input="y\n").output

def test_control_characters_from_a_repo_never_reach_the_terminal(gh, monkeypatch):
    from cpnlookup import llm
    from cpnlookup.retrieval import search
    gh.files["docs/x.md"] = "# Title \x1b]0;owned\x07 here\nbody about widgets\n"
    assert run("init", "o/r").exit_code == 0
    out = run("functions").output; assert "x: Title ]0;owned here" in out and "\x1b" not in out and "\x07" not in out
    monkeypatch.setattr(llm, "ensure_ready", lambda model: None); monkeypatch.setattr(llm, "stream_ollama", lambda prompt, **kw: iter(["fine"]))
    monkeypatch.setattr(search, "dense_search", lambda query, k: []); monkeypatch.setattr(search, "rerank", lambda query, cands, k: cands[:k])
    out = run("ask", "widgets?").output; assert "docs/x.md:1-2" in out and "\x1b" not in out and "\x07" not in out

def test_ask_reports_ollama_and_model_problems(ready, monkeypatch):
    from cpnlookup import llm
    from cpnlookup.indexer import embedder
    from cpnlookup.retrieval import search
    def offline(query, k): raise embedder.ModelError("Could not download the search model (x): offline. Check your connection and run the command again.")
    monkeypatch.setattr(search, "dense_search", offline)
    r = run("ask", "parse?"); assert r.exit_code == 1 and "Could not download the search model" in r.output and "Traceback" not in r.output
    def missing(model): raise llm.OllamaError(f"The model '{model}' is not installed in Ollama. Run: ollama pull {model}")
    monkeypatch.setattr(llm, "ensure_ready", missing)
    r = run("ask", "parse?"); assert r.exit_code == 1 and "ollama pull mistral" in r.output and not ready.prompts

def test_rerank_setting_and_30_index_suggestion(ready):
    from contextlib import closing
    from cpnlookup.indexer import storage
    assert "lookup init" not in run("ask", "parse?").output and ready.reranked == 1
    assert run("config", "rerank", "off").exit_code == 0
    with closing(storage.connect()) as conn: conn.execute("DELETE FROM repo_meta WHERE key = 'embed_model'"); conn.commit()
    r = run("ask", "parse?")
    assert r.exit_code == 0 and ready.reranked == 1 and "built by cpnlookup 3.0" in r.output and "ok" in r.output

def test_chat_answers_until_exit_and_survives_errors(ready):
    from cpnlookup import llm
    from cpnlookup.retrieval import memory
    r = run("chat", input="what is parse?\n\nand fetch_repo_tree?\nexit\nnever asked\n")
    assert r.exit_code == 0 and r.output.count("Sources:") == 2 and len(ready.prompts) == 2
    assert "Q: what is parse?" in ready.prompts[1]                                   # the first turn is context for the second
    ready.reply = [llm.OllamaError("Ollama stopped: boom")]
    r = run("chat", input="one?\n")                                                 # an error keeps the session open; end of input closes it
    assert r.exit_code == 0 and "boom" in r.output
    ready.reply = ["partial", KeyboardInterrupt()]
    r = run("chat", input="two?\nquit\n"); assert r.exit_code == 0 and "not saved" in r.output
    assert len(memory.get_all_history("o/r")) == 2

def test_terminal_stream_renders_markdown_and_keeps_partial_text(monkeypatch):
    import io
    from rich.console import Console
    from cpnlookup import ui
    out = io.StringIO(); monkeypatch.setattr(ui, "console", Console(file=out, force_terminal=True, width=60, height=8, color_system=None))
    text = ui.stream(iter(["# Title\n\n"] + [f"line {i}\n\n" for i in range(30)] + ["see [docs](http://evil.example/x)"]), "Asking m...")
    shown = out.getvalue()
    assert text.endswith("(http://evil.example/x)") and "line 29" in shown and "Title" in shown and "http://evil.example/x" in shown
    def broken():
        yield "kept so far"; raise RuntimeError("stream died")
    out.seek(0); out.truncate()
    with pytest.raises(RuntimeError): ui.stream(broken(), "Asking m...")
    assert "kept so far" in out.getvalue()

def test_piped_output_does_not_crash_on_legacy_windows_encoding(isolated):
    import os, subprocess, sys
    env = {**os.environ, "PYTHONIOENCODING": "cp1252", "MSYSTEM": "MINGW64"}         # what a Windows pipe gives by default; Git Bash reads UTF-8
    r = subprocess.run([sys.executable, "-c", "from cpnlookup.cli import cli; cli()", "auth", "tok"], capture_output=True, env=env)
    assert r.returncode == 0 and "\u2713 GitHub token saved." in r.stdout.decode("utf-8"), r.stderr

@pytest.mark.skipif(__import__("os").name != "nt", reason="console code pages are a Windows matter")
def test_piped_output_uses_the_console_code_page(monkeypatch):
    import ctypes, sys, types
    from cpnlookup import ui
    calls = []; fake = types.SimpleNamespace(encoding="cp1252", isatty=lambda: False, reconfigure=lambda **kw: calls.append(kw))
    monkeypatch.setattr(sys, "stdout", fake); monkeypatch.delenv("MSYSTEM", raising=False)
    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=types.SimpleNamespace(GetConsoleOutputCP=lambda: 437)))
    ui.fix_stdout(); assert calls == [{"encoding": "cp437", "errors": "replace"}]     # what PowerShell decodes a pipe with

def test_indexed_shows_the_version_that_built_each_index(gh):
    from contextlib import closing
    from cpnlookup.indexer import storage
    assert run("init", "o/r").exit_code == 0 and run("indexed").exit_code == 0
    assert storage.index_state(storage.db_path()) == (__version__, "[green]Active[/]")
    with closing(storage.connect()) as conn: conn.execute("DELETE FROM repo_meta WHERE key = 'cpnlookup_version'"); conn.commit()   # what 3.0 wrote
    assert storage.index_state(storage.db_path()) == ("3.0", "[yellow]Active (lookup init suggested)[/]")
    assert storage.index_state("nowhere/index.db") == ("-", "[red]Missing[/]")

def test_light_commands_import_nothing_heavy():
    import subprocess, sys
    code = ("import sys, cpnlookup.cli; print([m for m in ('numpy', 'requests', 'faiss', 'onnxruntime', 'tokenizers', 'huggingface_hub', 'torch', "
            "'sentence_transformers', 'markdown_it', 'rich.markdown') if m in sys.modules])")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() == "[]"

def test_profile_all_flag(monkeypatch):
    from cpnlookup import github
    seen = []
    def fake(user, limit): seen.append(limit); return [{"name": "r[1]", "stargazers_count": 3, "language": None}]
    monkeypatch.setattr(github, "get_user_repos", fake)
    assert "r[1]" in run("profile", "someone", "--all").output and "r[1]" in run("profile", "someone").output
    assert seen == [None, 20]
