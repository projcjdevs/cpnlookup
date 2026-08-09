import os
import click
import sqlite3
import subprocess
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["T_PGBAR"] = "0"
console = Console()

from cpnlookup.utils.config import save_github_token, get_local_config, save_local_config, update_registry, get_registry
from cpnlookup.github.client import get_user_repos
from cpnlookup.github.fetcher import fetch_repo_tree, fetch_file_contents
from cpnlookup.indexer.storage import (init_db, clear_local_index, get_local_db_path,
                                        set_index_status, get_index_status,
                                        load_file_hashes, delete_file_data,
                                        save_faiss_index)
from cpnlookup.llm.ollama import check_ollama, chat_with_ollama

def print_welcome_screen():
    header = """
 ██████╗██████╗ ███╗   ██╗██╗      ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗ 
██╔════╝██╔══██╗████╗  ██║██║     ██╔═══██╗██╔═══██╗██║ ██╔╝██║   ██║██╔══██╗
██║     ██████╔╝██╔██╗ ██║██║     ██║   ██║██║   ██║█████╔╝ ██║   ██║██████╔╝
██║     ██╔═══╝ ██║╚██╗██║██║     ██║   ██║██║   ██║██╔═██╗ ██║   ██║██╔═══╝ 
╚██████╗██║     ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██╗╚██████╔╝██║     
 ╚═════╝╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝   
 v2.0.0 - Forge: Consistent, Incremental, Reliable Indexing | by @projcjdevs  
    """
    console.print(f"[bold magenta]{header}[/]")
    ollama_status = "[bold green]Running[/]" if check_ollama() else "[bold red]Not Found (Required for 'ask')[/]"
    setup_info = f"""
[bold cyan]Quick Start:[/]
1. [white]Auth:[/] [green]lookup auth <token>[/]
2. [white]Index:[/] [green]lookup init <user/repo>[/]
3. [white]Ask:[/] [green]lookup ask "How does X work?"[/]

[bold cyan]Ollama Service:[/] {ollama_status}
[dim]Type [bold white]lookup help[/] for a detailed setup guide.[/]
[dim]Type [bold white]lookup commands[/] to see all available features.[/]  
[dim]Type [bold white]lookup desc[/] to see complete description of cpnlookup.[/] 
    """
    console.print(Panel(setup_info, border_style="magenta", box=box.ROUNDED))

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Local CLI tool for GitHub RAG and Codebase querying."""
    if ctx.invoked_subcommand is None: print_welcome_screen()

@cli.command()
def help():
    """In-depth guide for setting up and using cpnlookup."""
    ollama_panel = Panel(
        "1. Install Ollama from [bold cyan]ollama.com[/]\n"
        "2. Run [bold green]ollama serve[/] in a terminal.\n"
        "3. Download a model: [bold green]ollama pull mistral[/]",
        title="[bold white]Ollama Setup (The Brain)[/]", border_style="blue", box=box.ROUNDED
    )
    github_panel = Panel(
        "1. Go to GitHub [bold white]Settings > Developer Settings[/].\n"
        "2. Create a [bold white]Personal Access Token (Classic)[/].\n"
        "3. Select the [bold green]'repo'[/] scope.\n"
        "4. Run: [bold green]lookup auth <your_token>[/]",
        title="[bold white]GitHub Auth (The Data)[/]", border_style="green", box=box.ROUNDED
    )
    cmd_table = Table(box=box.SIMPLE, header_style="bold cyan")
    cmd_table.add_column("Command"); cmd_table.add_column("Purpose")
    cmd_table.add_row("desc", "Project info and version history")
    cmd_table.add_row("init", "Index a repo (Calculates Vectors + Graph)")
    cmd_table.add_row("ask", "Chat with the indexed codebase locally")
    cmd_table.add_row("indexed", "List all local indexes on this machine")
    cmd_table.add_row("drop", "Delete an index to save space")
    console.print(ollama_panel); console.print(github_panel)
    console.print(Panel(cmd_table, title="[bold white]Feature Reference[/]", border_style="magenta"))

@cli.command()
def commands():
    """List all available commands in a structured grid."""
    table = Table(title="Available Commands", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Command", style="cyan"); table.add_column("Description", style="white")
    cmds = [("auth", "Save GitHub Token"), ("profile", "List user repos"), ("init", "Index repository"),
            ("functions", "List code logic"), ("ask", "Hybrid RAG Query"),
            ("indexed", "Global registry"), ("config", "Set model/top_k"),
            ("clone", "Git clone repo"), ("drop", "Delete local index"),
            ("history", "View conversation history"),
            ("forget",  "Clear conversation memory"),]
    for c, d in cmds: table.add_row(c, d)
    console.print(table)

@cli.command()
def desc():
    """Project description, version history, and acknowledgements."""
    from rich.text import Text

    about = Panel(
        "[white]cpnlookup[/] is a local-first CLI tool for indexing and querying GitHub\n"
        "repositories through natural language — without ever running [green]git clone[/].\n\n"
        "It combines [bold cyan]FAISS vector similarity search[/] with a [bold cyan]SQLite-backed\n"
        "static call graph[/] to form a [bold magenta]Hybrid RAG pipeline[/] that understands\n"
        "both the [italic]meaning[/] and [italic]structure[/] of a codebase. All inference runs\n"
        "locally via [bold]Ollama[/]. No code, queries, or embeddings leave your machine.",
        title="[bold white]What is cpnlookup?[/]",
        border_style="magenta", box=box.ROUNDED
    )

    log_table = Table(box=box.SIMPLE, header_style="bold cyan", show_edge=False)
    log_table.add_column("Version", style="bold magenta", no_wrap=True)
    log_table.add_column("Name", style="bold white", no_wrap=True)
    log_table.add_column("Changes", style="dim white")
    log_table.add_row("v1.0.0", "Initial Release",
                      "Core Hybrid RAG pipeline. GitHub API indexing, AST chunking,\n"
                      "FAISS vector search, SQLite call graph, Ollama inference.")
    log_table.add_row("v1.1.1", "Optimized Indexing",
                      "Pre-index filtering (skip build artifacts, keep tests).\n"
                      "Batched embedding. Fixed missing runtime deps in pyproject.toml.")
    log_table.add_row("v1.2.2", "Lazy Imports",
                      "Heavy ML imports moved inside command functions.\n"
                      "Commands that don't need the model now start instantly.")
    log_table.add_row("v2.0.0", "Forge",
                      "Write consistency — index_status flag prevents corrupt state.\n"
                      "Incremental indexing — only changed files are re-fetched,\n"
                      "re-chunked, and re-embedded. Embedding BLOBs stored in SQLite.\n"
                      "Bidirectional graph traversal (callers + callees). faiss_id\n"
                      "column decouples FAISS positions from SQLite auto-increment IDs.")
    log_table.add_row("v3.0.0-beta", "Flux",
                      "Hybrid retrieval: BM25 sparse search merged with FAISS dense\n"
                      "search. Cross-encoder reranking for precision. Conversation\n"
                      "memory across sessions (per-repo SQLite). lookup history and\n"
                      "lookup forget commands.")
    version_panel = Panel(log_table, title="[bold white]Version History[/]",
                          border_style="cyan", box=box.ROUNDED)

    ack_text = (
        "[bold white]Developer[/]\n"
        "  [bold magenta]@projcjdevs[/] — architecture, implementation, and maintenance\n\n"
        "[bold white]Special Thanks[/]\n"
        "  [bold cyan]@2nieGarcia[/]       — feedback and testing\n"
        "  [bold cyan]@renzv-compsci[/]    — feedback and testing\n"
        "  [bold cyan]@NIghtIngale340[/]   — feedback and testing"
    )
    ack_panel = Panel(ack_text, title="[bold white]Acknowledgements[/]",
                      border_style="green", box=box.ROUNDED)

    console.print(about)
    console.print(version_panel)
    console.print(ack_panel)

@cli.command()
@click.argument('token')
def auth(token: str):
    save_github_token(token)
    console.print("[bold green]✓[/] GitHub token saved successfully.")

@cli.command()
@click.argument('username')
@click.argument('scope', required=False, default='default')
def profile(username: str, scope: str):
    with console.status(f"[bold cyan]Fetching {username}..."):
        try: repos = get_user_repos(username)
        except Exception as e: console.print(f"[red]Error:[/] {e}"); return
    if not repos: return
    show_all = scope.lower() == 'all'
    display = repos if show_all else repos[:20]
    table = Table(title=f"{username}'s Repos", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("#", style="dim"); table.add_column("Name"); table.add_column("Stars"); table.add_column("Language")
    for i, r in enumerate(display, 1): table.add_row(str(i), r.get("name"), str(r.get("stargazers_count")), r.get("language") or "N/A")
    console.print(table)

@cli.command()
@click.argument('repo_name')
def init(repo_name: str):
    from cpnlookup.indexer.chunker import chunk_python_code, chunk_markdown
    from cpnlookup.indexer.embedder import embed_chunks

    console.print(f"[bold cyan]Initializing {repo_name}...[/]")
    init_db()
    set_index_status('pending')

    with console.status("[bold yellow]Analyzing repository..."):
        try:
            db_path = get_local_db_path()
            tree = fetch_repo_tree(repo_name)       
            paths_in_repo = {f['path'] for f in tree}

            existing_hashes = load_file_hashes()      
            to_fetch  = [f for f in tree if existing_hashes.get(f['path']) != f['sha']]
            to_delete = [p for p in existing_hashes if p not in paths_in_repo]

            faiss_exists = (Path.cwd() / ".cpnlookup" / "faiss.index").exists()
            if not to_fetch and not to_delete and faiss_exists:
                set_index_status('complete')
                console.print("[bold green]✓[/] Index is already up to date."); return

            is_fresh = not existing_hashes 

            if to_delete or to_fetch:
                conn = sqlite3.connect(db_path); cursor = conn.cursor()
                for path in to_delete + [f['path'] for f in to_fetch]:
                    delete_file_data(cursor, path)
                conn.commit(); conn.close()

            new_files = fetch_file_contents(repo_name, to_fetch)

            if new_files:
                conn = sqlite3.connect(db_path); cursor = conn.cursor()
                new_chunks_map = {}  
                for f in new_files:
                    lang = 'python' if f['path'].endswith('.py') else 'markdown'
                    cursor.execute("INSERT OR REPLACE INTO raw_files (file_path, language, content, size_bytes, file_hash) VALUES (?, ?, ?, ?, ?)",
                                   (f['path'], lang, f['content'], f['size'], f['sha']))
                    chunks = []
                    if f['path'].endswith('.py'): chunks = chunk_python_code(f['path'], f['content'])
                    elif f['path'].lower().endswith('.md'): chunks = chunk_markdown(f['path'], f['content'])
                    new_chunks_map[f['path']] = chunks
                    for c in chunks:
                        cursor.execute("INSERT INTO chunks (name, file_path, line_start, line_end, chunk_type, source_code, docstring) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                       (c['name'], c['file_path'], c['line_start'], c['line_end'], c['chunk_type'], c['source_code'], c['docstring']))
                        cursor.execute("INSERT INTO graph_nodes (chunk_id, name, file_path) VALUES (?, ?, ?)",
                                       (cursor.lastrowid, c['name'], c['file_path']))
                conn.commit()

                cursor.execute("SELECT id, name FROM graph_nodes")
                all_nodes = cursor.fetchall(); all_names = {n[1] for n in all_nodes}
                for file_path, chunks in new_chunks_map.items():
                    for c in chunks:
                        if c['chunk_type'] == 'documentation': continue
                        cursor.execute("SELECT id FROM graph_nodes WHERE name = ? AND file_path = ?", (c['name'], c['file_path']))
                        node_row = cursor.fetchone()
                        if not node_row: continue
                        for target in all_names:
                            if target != c['name'] and f"{target}(" in c['source_code']:
                                cursor.execute("INSERT INTO graph_edges (source_id, target_name, edge_type) VALUES (?, ?, ?)",
                                               (node_row[0], target, "calls"))
                conn.commit(); conn.close()

            conn = sqlite3.connect(db_path); cursor = conn.cursor()
            cursor.execute("SELECT id, name, file_path, chunk_type, docstring, source_code, embedding FROM chunks ORDER BY id")
            all_rows = cursor.fetchall(); conn.close()

            if all_rows:
                new_file_paths = {f['path'] for f in to_fetch}

                needs_embed = [(r[0], {"name": r[1], "file_path": r[2], "chunk_type": r[3],
                                        "docstring": r[4], "source_code": r[5]})
                               for r in all_rows if r[2] in new_file_paths or r[6] is None]

                if needs_embed:
                    ids, dicts = zip(*needs_embed)
                    computed = embed_chunks(list(dicts))
                    conn = sqlite3.connect(db_path); cursor = conn.cursor()
                    for chunk_id, emb in zip(ids, computed):
                        cursor.execute("UPDATE chunks SET embedding = ? WHERE id = ?",
                                       (emb.astype(np.float32).tobytes(), chunk_id))
                    conn.commit(); conn.close()

                conn = sqlite3.connect(db_path); cursor = conn.cursor()
                cursor.execute("SELECT id, embedding FROM chunks ORDER BY id")
                embed_rows = cursor.fetchall(); conn.close()

                all_embeddings = np.array([np.frombuffer(r[1], dtype=np.float32) for r in embed_rows])
                save_faiss_index(all_embeddings)
                conn = sqlite3.connect(db_path); cursor = conn.cursor()
                for faiss_pos, row in enumerate(embed_rows):
                    cursor.execute("UPDATE chunks SET faiss_id = ? WHERE id = ?", (faiss_pos, row[0]))
                conn.commit(); conn.close()

                update_registry(os.getcwd(), repo_name, add=True)

            set_index_status('complete')

            total = len(all_rows) if all_rows else 0
            if is_fresh:
                console.print(f"[bold green]✓[/] Indexed {total} logic/doc units.")
            else:
                console.print(f"[bold green]✓[/] Re-indexed: {len(to_fetch)} changed, {len(to_delete)} removed, {total} total chunks.")

        except Exception as e:
            console.print(f"[red]Error:[/] {e}")

@cli.command()
@click.argument('question')
def ask(question: str):
    from cpnlookup.retrieval.vector_search import search_chunks
    from cpnlookup.retrieval.hybrid import bm25_search, merge_results, rerank
    from cpnlookup.retrieval.memory import get_recent_turns, save_turn
 
    db_path = get_local_db_path()
    if not db_path.exists(): console.print("[red]No index found.[/]"); return
    if get_index_status() == 'pending':
        console.print("[yellow]Index is incomplete. Run [bold]lookup init[/] again.[/]"); return
    if not check_ollama(): console.print("[red]Ollama not running.[/]"); return
 
    cfg = get_local_config()
    model  = cfg.get("model", "mistral")
    top_k  = cfg.get("top_k", 5)
    reg = get_registry()
    repo_name = reg.get(os.getcwd(), "unknown")
 
    with console.status(f"[cyan]Querying {model}..."):
        conn = sqlite3.connect(db_path); cursor = conn.cursor()
        cursor.execute(
            "SELECT faiss_id, name, file_path, chunk_type, docstring, source_code FROM chunks ORDER BY faiss_id"
        )
        rows = cursor.fetchall(); conn.close()

        all_chunks = [
            {"faiss_id": r[0], "name": r[1], "file_path": r[2],
             "chunk_type": r[3], "docstring": r[4], "source_code": r[5]}
            for r in rows if r[0] is not None
        ]
        faiss_ids   = search_chunks(question, top_k=top_k * 2) 
        bm25_hits   = bm25_search(question, all_chunks, top_k=top_k * 2)
        candidates  = merge_results(faiss_ids, bm25_hits, all_chunks)
        top_chunks  = rerank(question, candidates, top_k=top_k)
        conn = sqlite3.connect(db_path); cursor = conn.cursor()
        context, seen = [], set()
 
        for chunk in top_chunks:
            name, path, code = chunk['name'], chunk['file_path'], chunk['source_code']
            if name not in seen:
                context.append(f"--- FILE: {path} | NODE: {name} ---\n{code}")
                seen.add(name)
            cursor.execute("SELECT id FROM chunks WHERE name = ? AND file_path = ?", (name, path))
            id_row = cursor.fetchone()
            if not id_row: continue
            c_id = id_row[0]
 
            cursor.execute(
                "SELECT target_name FROM graph_edges "
                "WHERE source_id = (SELECT id FROM graph_nodes WHERE chunk_id = ?)", (c_id,)
            )
            for (n_name,) in cursor.fetchall():
                if n_name not in seen:
                    cursor.execute("SELECT file_path, source_code FROM chunks WHERE name = ?", (n_name,))
                    n_row = cursor.fetchone()
                    if n_row:
                        context.append(f"--- NEIGHBOR (callee): {n_row[0]} | {n_name} ---\n{n_row[1]}")
                        seen.add(n_name)
 
            cursor.execute("""
                SELECT gn.chunk_id FROM graph_edges ge
                JOIN graph_nodes gn ON ge.source_id = gn.id
                WHERE ge.target_name = (SELECT name FROM chunks WHERE id = ?)
            """, (c_id,))
            for (caller_chunk_id,) in cursor.fetchall():
                cursor.execute(
                    "SELECT name, file_path, source_code FROM chunks WHERE id = ?", (caller_chunk_id,)
                )
                caller = cursor.fetchone()
                if caller and caller[0] not in seen:
                    context.append(f"--- NEIGHBOR (caller): {caller[1]} | {caller[0]} ---\n{caller[2]}")
                    seen.add(caller[0])
 
        conn.close()

        recent = get_recent_turns(repo_name, n=4)
        memory_block = ""
        if recent:
            memory_block = "\n\nPrevious conversation:\n"
            for q, a in recent:
                memory_block += f"Q: {q}\nA: {a}\n\n"

        prompt = (
            "You are a code assistant. Use the provided code context to answer the question.\n\n"
            f"Code context:\n" + "\n\n".join(context) +
            memory_block +
            f"\n\nQuestion: {question}"
        )
        answer = chat_with_ollama(prompt, model=model)

        save_turn(repo_name, question, answer)
 
    console.print(f"\n[bold magenta]Q:[/] {question}\n" + "-"*30 + f"\n{answer}")
 
 
@cli.command()
def history():
    """Show conversation history for the current indexed repository."""
    from cpnlookup.retrieval.memory import get_all_history
    reg = get_registry()
    repo_name = reg.get(os.getcwd(), None)
    if not repo_name:
        console.print("[yellow]No indexed repository found in this directory.[/]"); return
 
    turns = get_all_history(repo_name)
    if not turns:
        console.print(f"[dim]No conversation history for [bold]{repo_name}[/].[/]"); return
 
    table = Table(title=f"History — {repo_name}", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("#",         style="dim",    no_wrap=True)
    table.add_column("Timestamp", style="dim",    no_wrap=True)
    table.add_column("Question",  style="cyan",   no_wrap=False)
    table.add_column("Answer",    style="white",  no_wrap=False)
 
    for t in turns:
        ts = t['timestamp'][:16].replace("T", " ")  # trim to YYYY-MM-DD HH:MM
        # Truncate long answers for display; full text is in the db.
        answer_preview = t['answer'][:120] + "..." if len(t['answer']) > 120 else t['answer']
        table.add_row(str(t['id']), ts, t['question'], answer_preview)
 
    console.print(table)
 
 
@cli.command()
@click.option('--all', 'wipe_all', is_flag=True, default=False,
              help="Clear memory for ALL repositories, not just the current one.")
def forget(wipe_all: bool):
    """Clear conversation memory for the current repository (or all repos with --all)."""
    from cpnlookup.retrieval.memory import clear_memory, clear_all_memory
    if wipe_all:
        if click.confirm("Delete ALL conversation history across every repository?"):
            n = clear_all_memory()
            console.print(f"[green]✓[/] Cleared {n} conversation turns.")
    else:
        reg = get_registry()
        repo_name = reg.get(os.getcwd(), None)
        if not repo_name:
            console.print("[yellow]No indexed repository found in this directory.[/]"); return
        if click.confirm(f"Delete conversation history for [bold]{repo_name}[/]?"):
            n = clear_memory(repo_name)
            console.print(f"[green]✓[/] Cleared {n} conversation turns for {repo_name}.")

@cli.command()
def functions():
    db_path = get_local_db_path()
    if not db_path.exists(): return
    if get_index_status() == 'pending':
        console.print("[yellow]Index is incomplete. Run [bold]lookup init[/] again.[/]"); return
    conn = sqlite3.connect(db_path); cursor = conn.cursor()
    cursor.execute("SELECT name, file_path, chunk_type FROM chunks ORDER BY file_path")
    rows = cursor.fetchall(); conn.close()
    table = Table(title="Indexed Logic", box=box.ROUNDED)
    table.add_column("Type", style="dim"); table.add_column("Name", style="cyan"); table.add_column("Path", style="green")
    for n, p, t in rows: table.add_row(t, n, p)
    console.print(table)

@cli.command()
def indexed():
    reg = get_registry()
    if not reg: console.print("[yellow]No indexes found.[/]"); return
    table = Table(title="Global Registry", box=box.ROUNDED)
    table.add_column("Repo", style="cyan"); table.add_column("Path", style="green"); table.add_column("Status")
    for p, r in reg.items():
        status = "[green]Active[/]" if os.path.exists(os.path.join(p, ".cpnlookup")) else "[red]Missing[/]"
        table.add_row(r, p, status)
    console.print(table)

@cli.command()
@click.argument('key')
@click.argument('value')
def config(key, value):
    cfg = get_local_config()
    if key == "top_k": value = int(value)
    cfg[key] = value
    save_local_config(cfg); console.print(f"[green]✓[/] Updated {key}.")

@cli.command()
@click.argument('repo_name')
def clone(repo_name: str):
    try:
        subprocess.run(["git", "clone", f"https://github.com/{repo_name}.git"], check=True)
        console.print("[green]✓[/] Cloned.")
    except Exception as e: console.print(f"[red]Error:[/] {e}")

@cli.command()
def drop():
    if click.confirm("Delete index?"):
        update_registry(os.getcwd(), "", add=False)
        if clear_local_index(): console.print("[green]✓[/] Deleted.")