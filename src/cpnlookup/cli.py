import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["T_PGBAR"] = "0" 

import click
import sqlite3
from rich.console import Console
from rich.table import Table
from rich import box 
from cpnlookup.utils.config import save_github_token
from cpnlookup.github.client import get_user_repos
from cpnlookup.github.fetcher import fetch_repo_files
from cpnlookup.indexer.storage import init_db, clear_local_index, get_local_db_path
from cpnlookup.indexer.chunker import chunk_python_code

console = Console()

@click.group()
def cli():
    """
    \b
       ██████╗██████╗  ███╗      ██╗██╗      ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗ 
    ██╔════╝██╔══██╗████╗  ██║██║     ██╔═══██╗██╔═══██╗██║ ██╔╝██║   ██║██╔══██╗
    ██║     ██████╔╝██╔██╗ ██║██║     ██║   ██║██║   ██║█████╔╝ ██║   ██║██████╔╝
    ██║     ██╔═══╝ ██║╚██╗██║██║     ██║   ██║██║   ██║██╔═██╗ ██║   ██║██╔═══╝ 
    ╚██████╗██║     ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██╗╚██████╔╝██║     
     ╚═════╝╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     

    Local CLI tool for GitHub RAG, Call Graphs, and Codebase querying.
    """
    pass

@cli.command()
@click.argument('token')
def auth(token: str):
    save_github_token(token)
    console.print("[bold green]✓[/bold green] GitHub token saved successfully to ~/.cpnlookup/auth.json")

@cli.command()
@click.argument('username')
@click.argument('scope', required=False, default='default')
def profile(username: str, scope: str):
    """List repos for a GitHub user. Use 'all' to show everything."""
    with console.status(f"[bold cyan]Fetching repos for {username}..."):
        try:
            repos = get_user_repos(username)
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            return
            
    if not repos:
        console.print(f"[yellow]No repositories found for {username}.[/]")
        return

    show_all = scope.lower() == 'all'
    limit = len(repos) if show_all else 20
    display_repos = repos[:limit]

    table = Table(
        title=f"\n[bold]{username}'s Repositories[/]", 
        box=box.ROUNDED,
        header_style="bold magenta",
        title_style="bold cyan"
    )
    table.add_column("#", style="dim", justify="center")
    table.add_column("Repo Name", style="white")
    table.add_column("Stars", justify="right", style="yellow")
    table.add_column("Language", style="green")
    
    for i, repo in enumerate(display_repos, 1):
        table.add_row(
            str(i),
            repo.get("name", "Unknown"),
            str(repo.get("stargazers_count", 0)),
            repo.get("language") or "N/A"
        )
        
    console.print(table)

    if not show_all and len(repos) > 20:
        remaining = len(repos) - 20
        console.print(f" [dim]... and {remaining} more repositories.[/]")
        console.print(f" [bold cyan]Tip:[/] Run [italic]lookup profile {username} all[/] to see the full list.\n")

@cli.command()
@click.argument('repo_name')
def init(repo_name: str):
    console.print(f"[bold cyan]Initializing local workspace for {repo_name}...[/]")
    init_db()
    
    with console.status("[bold yellow]Downloading codebase from GitHub..."):
        try:
            files = fetch_repo_files(repo_name)
            db_path = get_local_db_path()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for f in files:
                cursor.execute("""
                    INSERT OR IGNORE INTO raw_files (file_path, language, content, size_bytes)
                    VALUES (?, ?, ?, ?)
                """, (f['path'], 'python' if f['path'].endswith('.py') else None, f['content'], f['size']))
            
            console.print(f"[bold green]✓[/] Downloaded {len(files)} files.")
            
            console.print("[bold yellow]Chunking code into functions and classes...[/]")
            chunk_count = 0
            for f in files:
                if f['path'].endswith('.py'):
                    chunks = chunk_python_code(f['path'], f['content'])
                    for c in chunks:
                        cursor.execute("""
                            INSERT INTO chunks (name, file_path, line_start, line_end, chunk_type, source_code, docstring)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (c['name'], c['file_path'], c['line_start'], c['line_end'], c['chunk_type'], c['source_code'], c['docstring']))
                        chunk_count += 1

            conn.commit()
            conn.close()

            # --- START AI EMBEDDING ---
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, file_path, chunk_type, docstring FROM chunks")
            chunk_rows = cursor.fetchall()
            conn.close()

            if chunk_rows:
                console.print("\n[bold yellow]Generating AI vector index...[/]")
                
                chunks_for_ai = [
                    {"name": r[0], "file_path": r[1], "chunk_type": r[2], "docstring": r[3]} 
                    for r in chunk_rows
                ]
                
                from cpnlookup.indexer.embedder import embed_chunks
                from cpnlookup.indexer.storage import save_faiss_index
                
                embeddings = embed_chunks(chunks_for_ai)
                save_faiss_index(embeddings)
                
                console.print("[bold green]✓[/] AI Vector Index built successfully.")

            console.print(f"[bold green]✓[/] Successfully indexed {chunk_count} functions/classes.")
        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")

@cli.command()
def functions():
    db_path = get_local_db_path()
    if not db_path.exists():
        console.print("[red]No index found. Run 'lookup init <repo>' first.[/]")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name, file_path, chunk_type FROM chunks ORDER BY file_path")
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No functions indexed yet.[/]")
        return

    table = Table(title="Indexed Functions & Classes")
    table.add_column("Type", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("File Path", style="green")

    for name, path, ctype in rows:
        table.add_row(ctype, name, path)

    console.print(table)

@cli.command()
def drop():
    if click.confirm("Are you sure you want to delete the local index in this directory?"):
        if clear_local_index():
            console.print("[bold green]✓ Local index dropped successfully.[/]")
        else:
            console.print("[yellow]No local .cpnlookup folder found here.[/]")