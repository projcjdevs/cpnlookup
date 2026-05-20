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
from cpnlookup.retrieval.vector_search import search_chunks
from cpnlookup.llm.ollama import check_ollama, chat_with_ollama

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

# auth Command

@cli.command()
@click.argument('token')
def auth(token: str):
    save_github_token(token)
    console.print("[bold green]✓[/bold green] GitHub token saved successfully to ~/.cpnlookup/auth.json")

# profile Command

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

# init Command

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
                        cursor.execute("""
                            INSERT INTO graph_nodes (chunk_id, name, file_path)
                            VALUES (?, ?, ?)
                        """, (chunk_id, c['name'], c['file_path']))

                    # --- START GRAPH EDGE CONSTRUCTION ---
                        console.print("[bold yellow]Building call graph edges...[/]")
                        cursor.execute("SELECT id, name FROM graph_nodes")
                        nodes = cursor.fetchall()
                        all_names = {n[1] for n in nodes}

                        for node_id, node_name in nodes:
                             # Fetch the source code for this node
                            cursor.execute("SELECT source_code FROM chunks WHERE name = ?", (node_name,))
                            source = cursor.fetchone()[0]
                
                            for target_name in all_names:
                                if target_name != node_name and f"{target_name}(" in source:
                                    cursor.execute("""
                                        INSERT INTO graph_edges (source_id, target_name, edge_type)
                                        VALUES (?, ?, ?)
                                    """, (node_id, target_name, "calls"))
                    # --- END GRAPH EDGE CONSTRUCTION ---

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

# functions Command

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

# drop Command

@cli.command()
def drop():
    if click.confirm("Are you sure you want to delete the local index in this directory?"):
        if clear_local_index():
            console.print("[bold green]✓ Local index dropped successfully.[/]")
        else:
            console.print("[yellow]No local .cpnlookup folder found here.[/]")

# ask Command

@cli.command()
@click.argument('question')
def ask(question: str):
    db_path = get_local_db_path()
    if not db_path.exists():
        console.print("[red]No index found. Run 'lookup init' first.[/]")
        return

    from cpnlookup.llm.ollama import check_ollama, chat_with_ollama
    if not check_ollama():
        console.print("[bold red]Error:[/] Ollama is not running.")
        return

    from cpnlookup.utils.config import get_local_config
    cfg = get_local_config()
    model_name = cfg.get("model", "mistral")
    top_k = cfg.get("top_k", 5)

    with console.status(f"[bold cyan]Querying {model_name} with Hybrid RAG..."):
        from cpnlookup.retrieval.vector_search import search_chunks
        relevant_ids = search_chunks(question, top_k=top_k)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        context_blocks = []
        seen_names = set()

        for idx in relevant_ids:
            cursor.execute("""
                SELECT id, name, file_path, source_code 
                FROM chunks WHERE id = ?
            """, (idx + 1,))
            row = cursor.fetchone()
            
            if row:
                c_id, name, path, code = row
                if name not in seen_names:
                    context_blocks.append(f"--- FILE: {path} | NODE: {name} ---\n{code}")
                    seen_names.add(name)

                # Graph Expansion: Find functions this node calls
                cursor.execute("""
                    SELECT target_name FROM graph_edges 
                    WHERE source_id = (SELECT id FROM graph_nodes WHERE chunk_id = ?)
                """, (c_id,))
                neighbors = cursor.fetchall()
                
                for (n_name,) in neighbors:
                    if n_name not in seen_names:
                        cursor.execute("SELECT file_path, source_code FROM chunks WHERE name = ?", (n_name,))
                        n_row = cursor.fetchone()
                        if n_row:
                            context_blocks.append(f"--- NEIGHBOR (Called by {name}): {n_row[0]} | {n_name} ---\n{n_row[1]}")
                            seen_names.add(n_name)
        
        conn.close()
        
        context_text = "\n\n".join(context_blocks)
        prompt = f"""
        You are a technical assistant analyzing a codebase.
        Use the following retrieved code context to answer the user's question.
        
        CONTEXT:
        {context_text}

        USER QUESTION:
        {question}

        FINAL ANSWER:
        """
        
        answer = chat_with_ollama(prompt, model=model_name)

    console.print(f"\n[bold magenta]Question:[/] {question}")
    console.print(f"[dim]Context size: {len(seen_names)} functions/classes analyzed.[/]")
    console.print("-" * 30)
    console.print(answer)

# config Command

@cli.command()
@click.argument('key')
@click.argument('value')
def config(key, value):
    """Change local settings: lookup config set model llama3"""
    from cpnlookup.utils.config import get_local_config, save_local_config
    
    cfg = get_local_config()
    # Handle numeric values for top_k
    if key == "top_k":
        value = int(value)
        
    cfg[key] = value
    save_local_config(cfg)
    console.print(f"[green]✓[/] Config updated: {key} = {value}")