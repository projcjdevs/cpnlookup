"""One question from start to finish. Shared by `lookup ask` and `lookup chat`."""
from contextlib import closing
from cpnlookup import ui
from cpnlookup.indexer import storage

def ask(question: str, cfg: dict, repo: str) -> None:
    """Retrieves context, streams the answer live, saves the turn, lists the sources and says how long each part took.
    Raises OllamaError or ModelError; an answer that failed, was interrupted or came back empty is never saved."""
    from cpnlookup.llm import stream_ollama, OllamaError
    from cpnlookup.retrieval.search import retrieve
    from cpnlookup.retrieval.context import expand, build_prompt
    from cpnlookup.retrieval.memory import get_recent_turns, save_turn
    import time
    t0, first = time.perf_counter(), []
    def timed(pieces):
        for p in pieces: first or first.append(time.perf_counter()); yield p
    with ui.console.status("[cyan]Searching the index..."):
        with closing(storage.connect()) as conn: items = expand(conn, retrieve(conn, question, cfg["top_k"], cfg["rerank"] == "on"), cfg["max_neighbors"])
        prompt, used = build_prompt(question, items, get_recent_turns(repo, n=4), cfg["num_ctx"])
    t1 = time.perf_counter(); answer = ui.stream(timed(stream_ollama(prompt, model=cfg["model"], num_ctx=cfg["num_ctx"])), f"Asking {cfg['model']}..."); t2 = time.perf_counter()
    if not answer.strip(): raise OllamaError(f"{cfg['model']} returned an empty answer. Ask again, or pick another model with lookup config model <name>.")
    save_turn(repo, question, answer); ui.sources(used)
    ui.console.print(f"[dim]Took {t2 - t0:.1f} s: search {t1 - t0:.1f} s, waiting for the model {first[0] - t1:.1f} s, writing {t2 - first[0]:.1f} s[/]", highlight=False)
