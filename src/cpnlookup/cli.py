import os
from contextlib import closing
import click
from cpnlookup import __version__, ui
from cpnlookup.ui import console, esc, fail
from cpnlookup.config import save_github_token, get_local_config, set_local_config, update_registry, get_registry, local_dir
from cpnlookup.indexer.storage import connect, index_problem, current_repo, clear_local_index

@click.group(invoke_without_command=True)
@click.version_option(__version__, "-V", "--version", prog_name="cpnlookup")
@click.pass_context
def cli(ctx):
    """cpnlookup: ask questions about any GitHub repository, locally."""
    if ctx.invoked_subcommand is None: ui.welcome()

@cli.command()
def help():
    """Setup guide and the full command list."""
    ui.guide()

@cli.command(hidden=True)
def commands():
    """Alias for help."""
    ui.guide()

@cli.command()
def desc():
    """About cpnlookup, version history and acknowledgements."""
    ui.desc()

@cli.command()
@click.argument('token')
def auth(token: str):
    """Save a GitHub personal access token (repo scope)."""
    save_github_token(token); console.print("[bold green]✓[/] GitHub token saved.")

@cli.command()
@click.argument('username')
@click.option('--all', 'show_all', is_flag=True, help="List every repository, not just the 20 most recently updated.")
def profile(username: str, show_all: bool):
    """List a GitHub user's or organization's repositories."""
    from cpnlookup.github import get_user_repos, GitHubError
    with console.status(f"[bold cyan]Fetching {esc(username)}..."):
        try: repos = get_user_repos(username, limit=None if show_all else 20)
        except GitHubError as e: fail(esc(str(e)))
    if not repos: console.print(f"[yellow]{esc(username)} has no public repositories.[/]"); return
    table = ui.table(f"{esc(username)}'s repos", ("#", "dim"), ("Name", None), ("Stars", None), ("Language", None))
    for i, r in enumerate(repos, 1): table.add_row(str(i), esc(r["name"]), str(r.get("stargazers_count", 0)), esc(r.get("language") or "N/A"))
    console.print(table)

@cli.command()
@click.argument('repo_name')
def init(repo_name: str):
    """Index a GitHub repository (owner/repo or URL) into this directory. Re-runs only process changed files."""
    from cpnlookup.github import parse_repo, GitHubError
    from cpnlookup.indexer.pipeline import run_init
    try: repo = parse_repo(repo_name)
    except GitHubError as e: fail(esc(str(e)))
    console.print(f"[bold cyan]Initializing {esc(repo)}...[/]")
    with console.status("[bold yellow]Starting...") as status:
        try: r = run_init(repo, lambda m: status.update(f"[bold yellow]{esc(m)}"))
        except GitHubError as e: fail(esc(str(e)))
        except Exception as e: fail(f"{esc(str(e))}\n[dim]The index is marked incomplete; run lookup init again.[/]")
    if r["upgraded"]: console.print("[yellow]Rebuilt an index created by an older cpnlookup version.[/]")
    if r["truncated"]: console.print("[yellow]GitHub truncated the file tree for this very large repo; some files were not indexed.[/]")
    if r["up_to_date"]: console.print("[bold green]✓[/] Index is already up to date."); return
    if r["fresh"]: console.print(f"[bold green]✓[/] Indexed {r['total']} chunks with {r['edges']} call edges.")
    else: console.print(f"[bold green]✓[/] Re-indexed: {r['changed']} changed, {r['removed']} removed, {r['total']} chunks total.")

@cli.command()
@click.argument('question')
def ask(question: str):
    """Ask a question about the indexed repository."""
    from cpnlookup.llm import check_ollama, chat_with_ollama, OllamaError
    from cpnlookup.retrieval.search import retrieve
    from cpnlookup.retrieval.context import expand, build_prompt
    from cpnlookup.retrieval.memory import get_recent_turns, save_turn
    problem = index_problem()
    if problem: fail(problem)
    if not check_ollama(): fail("Ollama is not running. Start it with [bold]ollama serve[/].")
    cfg = get_local_config(); repo = current_repo() or "unknown"
    with console.status("[cyan]Searching the index (loading models)...") as status:
        with closing(connect()) as conn: items = expand(conn, retrieve(conn, question, cfg["top_k"]), cfg["max_neighbors"])
        prompt = build_prompt(question, items, get_recent_turns(repo, n=4), cfg["num_ctx"])
        status.update(f"[cyan]Asking {esc(cfg['model'])}...")
        try: answer = chat_with_ollama(prompt, model=cfg["model"], num_ctx=cfg["num_ctx"])
        except OllamaError as e: fail(esc(str(e)))
    save_turn(repo, question, answer)
    console.print(f"\n[bold magenta]Q:[/] {esc(question)}\n" + "-" * 30); console.print(answer, markup=False, highlight=False)

@cli.command()
def history():
    """Show conversation history for this directory's repository."""
    from cpnlookup.retrieval.memory import get_all_history
    repo = current_repo()
    if not repo: fail("No indexed repository in this directory.")
    turns = get_all_history(repo)
    if not turns: console.print(f"[dim]No conversation history for [bold]{esc(repo)}[/].[/]"); return
    table = ui.table(f"History — {esc(repo)}", ("#", "dim"), ("Time", "dim"), ("Question", "cyan"), ("Answer", "white"))
    for t in turns: table.add_row(str(t["id"]), t["timestamp"][:16].replace("T", " "), esc(t["question"]), esc(t["answer"][:120] + ("..." if len(t["answer"]) > 120 else "")))
    console.print(table)

@cli.command()
@click.option('--all', 'wipe_all', is_flag=True, help="Clear memory for every repository, not just this one.")
def forget(wipe_all: bool):
    """Clear conversation memory for this repository (or all with --all)."""
    from cpnlookup.retrieval.memory import clear_memory
    repo = None if wipe_all else current_repo()
    if not wipe_all and not repo: fail("No indexed repository in this directory.")
    if click.confirm("Delete ALL conversation history for every repository?" if wipe_all else f"Delete conversation history for {repo}?"):
        console.print(f"[green]✓[/] Cleared {clear_memory(repo)} conversation turns.")

@cli.command()
def functions():
    """List every indexed function, class and documentation section."""
    problem = index_problem()
    if problem: fail(problem)
    with closing(connect()) as conn: rows = conn.execute("SELECT chunk_type, name, file_path, line_start FROM chunks ORDER BY file_path, line_start").fetchall()
    table = ui.table(f"Indexed logic ({len(rows)})", ("Type", "dim"), ("Name", "cyan"), ("Location", "green"))
    for t, n, p, line in rows: table.add_row(t, esc(n), esc(f"{p}:{line}"))
    console.print(table)

@cli.command()
def indexed():
    """List every directory on this machine that has an index."""
    reg = get_registry()
    if not reg: console.print("[yellow]No indexes yet. Run lookup init <owner/repo> in a project directory.[/]"); return
    table = ui.table("Indexed repositories", ("Repo", "cyan"), ("Path", "green"), ("Status", None))
    for p, r in reg.items(): table.add_row(esc(r), esc(p), "[green]Active[/]" if os.path.exists(os.path.join(p, ".cpnlookup", "index.db")) else "[red]Missing[/]")
    console.print(table)

@cli.command()
@click.argument('key', required=False)
@click.argument('value', required=False)
def config(key, value):
    """Show settings, or set one: model, top_k, num_ctx, max_neighbors."""
    if key is None:
        table = ui.table("Settings for this directory", ("Key", "cyan"), ("Value", None))
        for k, v in get_local_config().items(): table.add_row(k, esc(str(v)))
        console.print(table); return
    if value is None: fail(f"Give a value: lookup config {esc(key)} <value>")
    try: console.print(f"[green]✓[/] {esc(key)} = {esc(str(set_local_config(key, value)))}")
    except ValueError as e: fail(esc(str(e)))

@cli.command()
@click.argument('repo_name')
def clone(repo_name: str):
    """git clone a GitHub repository into the current directory."""
    import subprocess
    from cpnlookup.github import parse_repo, GitHubError
    try: repo = parse_repo(repo_name)
    except GitHubError as e: fail(esc(str(e)))
    try: subprocess.run(["git", "clone", f"https://github.com/{repo}.git"], check=True); console.print("[green]✓[/] Cloned.")
    except FileNotFoundError: fail("git is not installed or not on PATH.")
    except subprocess.CalledProcessError as e: fail(f"git clone failed (exit {e.returncode}).")

@cli.command()
def drop():
    """Delete this directory's index. Conversation memory is kept."""
    if not local_dir().exists(): fail("No index in this directory.")
    if click.confirm("Delete the index in this directory?"): update_registry(os.getcwd(), "", add=False); clear_local_index(); console.print("[green]✓[/] Deleted.")
