import os
import click
import sqlite3
import json
import subprocess
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Environment Setup
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["T_PGBAR"] = "0" 

# Core Imports
from cpnlookup.utils.config import save_github_token, get_local_config, save_local_config, update_registry, get_registry
from cpnlookup.github.client import get_user_repos
from cpnlookup.github.fetcher import fetch_repo_files
from cpnlookup.indexer.storage import init_db, clear_local_index, get_local_db_path
from cpnlookup.indexer.chunker import chunk_python_code, chunk_markdown
from cpnlookup.retrieval.vector_search import search_chunks
from cpnlookup.llm.ollama import check_ollama, chat_with_ollama
from cpnlookup.output.mermaid import generate_mermaid_graph

console = Console()

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
    
    setup_info = """
[bold cyan]How to Setup & Use cpnlookup:[/]
1. [bold white]Auth:[/] Save your token: [green]lookup auth <your_token>[/]
2. [bold white]Browse:[/] Find a repo: [green]lookup profile <username>[/]
3. [bold white]Index:[/] Download and analyze: [green]lookup init <user/repo>[/]
4. [bold white]Query:[/] Chat with your code: [green]lookup ask "How does X work?"[/]

[dim]Type [bold white]lookup commands[/] to see all available features.[/]
    """
    console.print(Panel(setup_info, border_style="cyan", box=box.ROUNDED))

# Main Group Entrypoint
@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Local CLI tool for GitHub RAG, Call Graphs, and Codebase querying."""
    if ctx.invoked_subcommand is None:
        print_welcome_screen()

# commands Command
@cli.command()
def commands():
    """List all available commands and their descriptions."""
    table = Table(title="Available cpnlookup Commands", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    command_list = [
        ("auth", "Save your GitHub Personal Access Token."),
        ("profile", "List repositories for a GitHub user."),
        ("init", "Index a GitHub repository locally (Logic + Docs)."),
        ("status", "Show info about the current local index."),
        ("functions", "List all indexed Python functions and classes."),
        ("ask", "Ask natural language questions using Hybrid RAG."),
        ("mermaid", "Generate a Mermaid.js call graph diagram."),
        ("indexed", "Show all repositories indexed on this machine."),
        ("config", "Change local settings (model, top_k)."),
        ("clone", "Standard git clone of the indexed repository."),
        ("drop", "Delete the local index in the current directory.")
    ]

    for cmd, desc in command_list:
        table.add_row(cmd, desc)
    
    console.print(table)

# auth Command
@cli.command()
@click.argument('token')
def auth(token: str):
    save_github_token(token)
    console.print("[bold green]✓[/bold green] GitHub token saved successfully.")

# profile Command
@cli.command()
@click.argument('username')
@click.argument('scope', required=False, default='default')
def profile(username: str, scope: str):
    with console.status(f"[bold cyan]Fetching repos for {username}..."):
        try:
            repos = get_user_repos(username)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            return
            
    if not repos: return

    show_all = scope.lower() == 'all'
    limit = len(repos) if show_all else 20
    display_repos = repos[:limit]

    table = Table(title=f"\n[bold]{username}'s Repositories[/]", box=box.ROUNDED, header_style="bold magenta")
    table.add_column("#", style="dim", justify="center")
    table.add_column("Repo Name", style="white")
    table.add_column("Stars", justify="right", style="yellow")
    table.add_column("Language", style="green")
    
    for i, repo in enumerate(display_repos, 1):
        table.add_row(str(i), repo.get("name", "Unknown"), str(repo.get("stargazers_count", 0)), repo.get("language") or "N/A")
        
    console.print(table)
    if not show_all and len(repos) > 20:
        console.print(f" [dim]... and {len(repos)-20} more. Run [italic]lookup profile {username} all[/] to see all.[/]\n")

# init Command
@cli.command()
@click.argument('repo_name')
def init(repo_name: str):
    console.print(f"[bold cyan]Initializing local workspace for {repo_name}...[/]")
    init_db()
    
    with console.status("[bold yellow]Downloading codebase..."):
        try:
            files = fetch_repo_files(repo_name)
            db_path = get_local_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for f in files:
                lang = 'python' if f['path'].endswith('.py') else 'markdown'
                cursor.execute("INSERT OR IGNORE INTO raw_files (file_path, language, content, size_bytes) VALUES (?, ?, ?, ?)", 
                             (f['path'], lang, f['content'], f['size']))
            
            console.print(f"[bold green]✓[/] Downloaded {len(files)} files.")
            
            chunk_count = 0
            for f in files:
                chunks = []
                if f['path'].endswith('.py'):
                    chunks = chunk_python_code(f['path'], f['content'])
                elif f['path'].lower().endswith('.md'):
                    chunks = chunk_markdown(f['path'], f['content'])
                
                for c in chunks:
                    cursor.execute("INSERT INTO chunks (name, file_path, line_start, line_end, chunk_type, source_code, docstring) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                 (c['name'], c['file_path'], c['line_start'], c['line_end'], c['chunk_type'], c['source_code'], c['docstring']))
                    chunk_id = cursor.lastrowid
                    cursor.execute("INSERT INTO graph_nodes (chunk_id, name, file_path) VALUES (?, ?, ?)", (chunk_id, c['name'], c['file_path']))
                    chunk_count += 1

            cursor.execute("SELECT id, name FROM graph_nodes")
            nodes = cursor.fetchall()
            all_names = {n[1] for n in nodes}

            for node_id, node_name in nodes:
                cursor.execute("SELECT source_code, chunk_type FROM chunks WHERE name = ?", (node_name,))
                res = cursor.fetchone()
                if not res or res[1] == 'documentation': continue
                
                source = res[0]
                for target_name in all_names:
                    if target_name != node_name and f"{target_name}(" in source:
                        cursor.execute("INSERT INTO graph_edges (source_id, target_name, edge_type) VALUES (?, ?, ?)", (node_id, target_name, "calls"))

            conn.commit()
            conn.close()

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, file_path, chunk_type, docstring FROM chunks")
            chunk_rows = cursor.fetchall()
            conn.close()

            if chunk_rows:
                from cpnlookup.indexer.embedder import embed_chunks
                from cpnlookup.indexer.storage import save_faiss_index
                embeddings = embed_chunks([{"name": r[0], "file_path": r[1], "chunk_type": r[2], "docstring": r[3]} for r in chunk_rows])
                save_faiss_index(embeddings)
                
                update_registry(os.getcwd(), repo_name, add=True)
                console.print(f"[bold green]✓[/] AI Vector Index built & Repo registered globally.")

            console.print(f"[bold green]✓[/] Successfully indexed {chunk_count} logic/doc units.")
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")

# ask Command
@cli.command()
@click.argument('question')
def ask(question: str):
    db_path = get_local_db_path()
    if not db_path.exists():
        console.print("[red]No index found. Run 'lookup init' first.[/]")
        return

    if not check_ollama():
        console.print("[bold red]Error:[/] Ollama is not running.")
        return

    cfg = get_local_config()
    model_name = cfg.get("model", "mistral")
    top_k = cfg.get("top_k", 5)

    with console.status(f"[bold cyan]Querying {model_name}..."):
        relevant_ids = search_chunks(question, top_k=top_k)
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        context_blocks, seen_names = [], set()

        for idx in relevant_ids:
            cursor.execute("SELECT id, name, file_path, source_code FROM chunks WHERE id = ?", (idx + 1,))
            row = cursor.fetchone()
            if row:
                c_id, name, path, code = row
                if name not in seen_names:
                    context_blocks.append(f"--- FILE: {path} | NODE: {name} ---\n{code}")
                    seen_names.add(name)

                cursor.execute("SELECT target_name FROM graph_edges WHERE source_id = (SELECT id FROM graph_nodes WHERE chunk_id = ?)", (c_id,))
                for (n_name,) in cursor.fetchall():
                    if n_name not in seen_names:
                        cursor.execute("SELECT file_path, source_code FROM chunks WHERE name = ?", (n_name,))
                        n_row = cursor.fetchone()
                        if n_row:
                            context_blocks.append(f"--- NEIGHBOR (Called by {name}): {n_row[0]} | {n_name} ---\n{n_row[1]}")
                            seen_names.add(n_name)
        conn.close()
        
        prompt = f"Use this context to answer: {question}\n\nCONTEXT:\n" + "\n\n".join(context_blocks)
        answer = chat_with_ollama(prompt, model=model_name)

    console.print(f"\n[bold magenta]Q:[/] {question}\n[dim]Context size: {len(seen_names)} units analyzed.[/]\n" + "-"*30 + f"\n{answer}")

# functions Command
@cli.command()
def functions():
    db_path = get_local_db_path()
    if not db_path.exists(): return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, file_path, chunk_type FROM chunks ORDER BY file_path")
    rows = cursor.fetchall()
    conn.close()

    table = Table(title="Indexed logic", box=box.ROUNDED)
    table.add_column("Type", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("File Path", style="green")
    for name, path, ctype in rows: table.add_row(ctype, name, path)
    console.print(table)

# mermaid Command
@cli.command()
def mermaid():
    """Generate a Mermaid.js call graph diagram."""
    db_path = get_local_db_path()
    if not db_path.exists(): return
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, source_code FROM chunks")
    rows = cursor.fetchall()
    conn.close()

    chart = generate_mermaid_graph([{"name": r[0], "source_code": r[1]} for r in rows])
    console.print(Panel(chart, title="Mermaid Syntax (Paste into mermaid.live)", border_style="bright_blue"))

# indexed Command
@cli.command()
def indexed():
    registry = get_registry()
    if not registry: return
    table = Table(title="Globally Indexed Repositories", box=box.ROUNDED)
    table.add_column("Repository", style="cyan")
    table.add_column("Local Path", style="green")
    table.add_column("Status")
    for path, repo in registry.items():
        status = "[green]Active[/]" if os.path.exists(os.path.join(path, ".cpnlookup")) else "[red]Missing[/]"
        table.add_row(repo, path, status)
    console.print(table)

# config Command
@cli.command()
@click.argument('key')
@click.argument('value')
def config(key, value):
    cfg = get_local_config()
    if key == "top_k": value = int(value)
    cfg[key] = value
    save_local_config(cfg)
    console.print(f"[green]✓[/] Updated: {key} = {value}")

# clone Command
@cli.command()
@click.argument('repo_name')
def clone(repo_name: str):
    url = f"https://github.com/{repo_name}.git"
    try:
        subprocess.run(["git", "clone", url], check=True)
        console.print("[bold green]✓[/] Repository cloned.")
    except Exception as e: console.print(f"[red]Error:[/] {e}")

# drop Command
@cli.command()
def drop():
    if click.confirm("Delete the local index?"):
        update_registry(os.getcwd(), "", add=False)
        if clear_local_index(): console.print("[green]✓[/] Deleted.")