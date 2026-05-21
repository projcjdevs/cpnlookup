import os
import click
import sqlite3
import json
import subprocess
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box

# Environment & UI Setup
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["T_PGBAR"] = "0" 
console = Console()

# Core Module Imports
from cpnlookup.utils.config import save_github_token, get_local_config, save_local_config, update_registry, get_registry
from cpnlookup.github.client import get_user_repos
from cpnlookup.github.fetcher import fetch_repo_files
from cpnlookup.indexer.storage import init_db, clear_local_index, get_local_db_path
from cpnlookup.indexer.chunker import chunk_python_code, chunk_markdown
from cpnlookup.retrieval.vector_search import search_chunks
from cpnlookup.llm.ollama import check_ollama, chat_with_ollama

# Welcome UI logic
def print_welcome_screen():
    header = """
 ██████╗██████╗ ███╗   ██╗██╗      ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗ 
██╔════╝██╔══██╗████╗  ██║██║     ██╔═══██╗██╔═══██╗██║ ██╔╝██║   ██║██╔══██╗
██║     ██████╔╝██╔██╗ ██║██║     ██║   ██║██║   ██║█████╔╝ ██║   ██║██████╔╝
██║     ██╔═══╝ ██║╚██╗██║██║     ██║   ██║██║   ██║██╔═██╗ ██║   ██║██╔═══╝ 
╚██████╗██║     ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██╗╚██████╔╝██║     
 ╚═════╝╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝   

Local CLI tool for GitHub RAG, Call Graphs, and Codebase querying -- Made by @projcjdevs.

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
    """
    console.print(Panel(setup_info, border_style="magenta", box=box.ROUNDED))

# Main CLI Entrypoint
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Local CLI tool for GitHub RAG, Call Graphs, and Codebase querying."""
    if ctx.invoked_subcommand is None:
        print_welcome_screen()

# help Command (In-depth Guide)
@cli.command()
def help():
    """In-depth guide for setting up and using cpnlookup."""
    
    # Section 1: External Dependencies
    ollama_panel = Panel(
        "1. Install Ollama from [bold cyan]ollama.com[/]\n"
        "2. Run [bold green]ollama serve[/] in a terminal.\n"
        "3. Download a model: [bold green]ollama pull mistral[/]\n"
        "[dim]Note: cpnlookup defaults to mistral, but you can change it in config.[/]",
        title="[bold white]Ollama Setup (The Brain)[/]", border_style="blue", box=box.ROUNDED
    )

    # Section 2: GitHub Setup
    github_panel = Panel(
        "1. Go to GitHub [bold white]Settings > Developer Settings[/].\n"
        "2. Create a [bold white]Personal Access Token (Classic)[/].\n"
        "3. Select the [bold green]'repo'[/] scope.\n"
        "4. Run: [bold green]lookup auth <your_token>[/]",
        title="[bold white]GitHub Auth (The Data)[/]", border_style="green", box=box.ROUNDED
    )

    # Section 3: Commands Reference
    cmd_table = Table(box=box.SIMPLE, header_style="bold cyan")
    cmd_table.add_column("Command")
    cmd_table.add_column("Purpose")
    cmd_table.add_row("profile", "Browse a user's repositories.")
    cmd_table.add_row("init", "Index a repo (Calculates Vectors + Graph).")
    cmd_table.add_row("ask", "Chat with the indexed codebase locally.")
    cmd_table.add_row("indexed", "List all local indexes on this machine.")
    cmd_table.add_row("drop", "Delete an index to save space.")

    console.print(ollama_panel)
    console.print(github_panel)
    console.print(Panel(cmd_table, title="[bold white]Feature Reference[/]", border_style="magenta"))

# commands Command (Simplified List)
@cli.command()
def commands():
    """List all commands in a structured grid."""
    table = Table(title="Available Commands", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    cmds = [("auth", "Save GitHub Token"), ("profile", "List user repos"), ("init", "Index repository"), 
            ("status", "Show index info"), ("functions", "List code logic"), ("ask", "Hybrid RAG Query"),
            ("indexed", "Global registry"), ("config", "Set model/top_k"),
            ("clone", "Git clone repo"), ("drop", "Delete local index")]
    for c, d in cmds: table.add_row(c, d)
    console.print(table)

# auth Command
@cli.command()
@click.argument('token')
def auth(token: str):
    save_github_token(token)
    console.print("[bold green]✓[/] GitHub token saved successfully.")

# profile Command
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

# init Command
@cli.command()
@click.argument('repo_name')
def init(repo_name: str):
    console.print(f"[bold cyan]Initializing {repo_name}...[/]")
    init_db()
    with console.status("[bold yellow]Downloading and Analyzing..."):
        try:
            files = fetch_repo_files(repo_name)
            db_path = get_local_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            for f in files:
                lang = 'python' if f['path'].endswith('.py') else 'markdown'
                cursor.execute("INSERT OR IGNORE INTO raw_files (file_path, language, content, size_bytes) VALUES (?, ?, ?, ?)", 
                             (f['path'], lang, f['content'], f['size']))
            
            chunk_count = 0
            for f in files:
                chunks = []
                if f['path'].endswith('.py'): chunks = chunk_python_code(f['path'], f['content'])
                elif f['path'].lower().endswith('.md'): chunks = chunk_markdown(f['path'], f['content'])
                for c in chunks:
                    cursor.execute("INSERT INTO chunks (name, file_path, line_start, line_end, chunk_type, source_code, docstring) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                 (c['name'], c['file_path'], c['line_start'], c['line_end'], c['chunk_type'], c['source_code'], c['docstring']))
                    chunk_id = cursor.lastrowid
                    cursor.execute("INSERT INTO graph_nodes (chunk_id, name, file_path) VALUES (?, ?, ?)", (chunk_id, c['name'], c['file_path']))
                    chunk_count += 1

            cursor.execute("SELECT id, name FROM graph_nodes")
            nodes = cursor.fetchall(); all_names = {n[1] for n in nodes}
            for node_id, node_name in nodes:
                cursor.execute("SELECT source_code, chunk_type FROM chunks WHERE name = ?", (node_name,))
                res = cursor.fetchone()
                if not res or res[1] == 'documentation': continue
                for target in all_names:
                    if target != node_name and f"{target}(" in res[0]:
                        cursor.execute("INSERT INTO graph_edges (source_id, target_name, edge_type) VALUES (?, ?, ?)", (node_id, target, "calls"))
            conn.commit(); conn.close()

            conn = sqlite3.connect(db_path); cursor = conn.cursor()
            cursor.execute("SELECT name, file_path, chunk_type, docstring FROM chunks")
            rows = cursor.fetchall(); conn.close()
            if rows:
                from cpnlookup.indexer.embedder import embed_chunks
                from cpnlookup.indexer.storage import save_faiss_index
                embeddings = embed_chunks([{"name": r[0], "file_path": r[1], "chunk_type": r[2], "docstring": r[3]} for r in rows])
                save_faiss_index(embeddings)
                update_registry(os.getcwd(), repo_name, add=True)
            console.print(f"[bold green]✓[/] Successfully indexed {chunk_count} logic/doc units.")
        except Exception as e: console.print(f"[red]Error:[/] {e}")

# ask Command
@cli.command()
@click.argument('question')
def ask(question: str):
    db_path = get_local_db_path()
    if not db_path.exists(): console.print("[red]No index found.[/]"); return
    if not check_ollama(): console.print("[red]Ollama not running.[/]"); return
    cfg = get_local_config(); model = cfg.get("model", "mistral"); top_k = cfg.get("top_k", 5)
    with console.status(f"[cyan]Querying {model}..."):
        relevant_ids = search_chunks(question, top_k=top_k)
        conn = sqlite3.connect(db_path); cursor = conn.cursor()
        context, seen = [], set()
        for idx in relevant_ids:
            cursor.execute("SELECT id, name, file_path, source_code FROM chunks WHERE id = ?", (idx + 1,))
            row = cursor.fetchone()
            if row:
                c_id, name, path, code = row
                if name not in seen: context.append(f"--- FILE: {path} | NODE: {name} ---\n{code}"); seen.add(name)
                cursor.execute("SELECT target_name FROM graph_edges WHERE source_id = (SELECT id FROM graph_nodes WHERE chunk_id = ?)", (c_id,))
                for (n_name,) in cursor.fetchall():
                    if n_name not in seen:
                        cursor.execute("SELECT file_path, source_code FROM chunks WHERE name = ?", (n_name,))
                        n_row = cursor.fetchone()
                        if n_row: context.append(f"--- NEIGHBOR: {n_row[0]} | {n_name} ---\n{n_row[1]}"); seen.add(n_name)
        conn.close()
        answer = chat_with_ollama(f"Context:\n" + "\n\n".join(context) + f"\n\nQuestion: {question}", model=model)
    console.print(f"\n[bold magenta]Q:[/] {question}\n" + "-"*30 + f"\n{answer}")

# functions Command
@cli.command()
def functions():
    db_path = get_local_db_path()
    if not db_path.exists(): return
    conn = sqlite3.connect(db_path); cursor = conn.cursor()
    cursor.execute("SELECT name, file_path, chunk_type FROM chunks ORDER BY file_path")
    rows = cursor.fetchall(); conn.close()
    table = Table(title="Indexed Logic", box=box.ROUNDED)
    table.add_column("Type", style="dim"); table.add_column("Name", style="cyan"); table.add_column("Path", style="green")
    for n, p, t in rows: table.add_row(t, n, p)
    console.print(table)

# indexed Command
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

# config Command
@cli.command()
@click.argument('key')
@click.argument('value')
def config(key, value):
    cfg = get_local_config()
    if key == "top_k": value = int(value)
    cfg[key] = value
    save_local_config(cfg); console.print(f"[green]✓[/] Updated {key}.")

# clone Command
@cli.command()
@click.argument('repo_name')
def clone(repo_name: str):
    try:
        subprocess.run(["git", "clone", f"https://github.com/{repo_name}.git"], check=True)
        console.print("[green]✓[/] Cloned.")
    except Exception as e: console.print(f"[red]Error:[/] {e}")

# drop Command
@cli.command()
def drop():
    if click.confirm("Delete index?"):
        update_registry(os.getcwd(), "", add=False)
        if clear_local_index(): console.print("[green]✓[/] Deleted.")