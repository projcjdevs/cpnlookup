from rich import box
from rich.console import Console
from rich.markup import escape as esc
from rich.panel import Panel
from rich.table import Table
from cpnlookup import __version__

console = Console()
CODENAME = "Flux"
BANNER = r"""
 ██████╗██████╗ ███╗   ██╗██╗      ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗
██╔════╝██╔══██╗████╗  ██║██║     ██╔═══██╗██╔═══██╗██║ ██╔╝██║   ██║██╔══██╗
██║     ██████╔╝██╔██╗ ██║██║     ██║   ██║██║   ██║█████╔╝ ██║   ██║██████╔╝
██║     ██╔═══╝ ██║╚██╗██║██║     ██║   ██║██║   ██║██╔═██╗ ██║   ██║██╔═══╝
╚██████╗██║     ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██╗╚██████╔╝██║
 ╚═════╝╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝"""
COMMANDS = [("help", "Setup guide and this command list"), ("desc", "About, version history, acknowledgements"),
            ("auth <token>", "Save your GitHub personal access token"), ("profile <user> [--all]", "List a user's repositories"),
            ("init <owner/repo>", "Index a repository into this directory"), ('ask "<question>"', "Ask about the indexed code"),
            ("functions", "List indexed functions, classes and docs"), ("indexed", "List every index on this machine"),
            ("config [key] [value]", "Show or set model, top_k, num_ctx, max_neighbors"), ("clone <owner/repo>", "git clone a repository"),
            ("drop", "Delete this directory's index"), ("history", "Show conversation history"), ("forget [--all]", "Clear conversation memory")]

def fail(msg: str):
    """Prints an error (msg may contain markup; escape untrusted parts) and exits with status 1."""
    console.print(f"[bold red]Error:[/] {msg}"); raise SystemExit(1)

def table(title: str, *cols, **kw) -> Table:
    t = Table(title=title, box=box.ROUNDED, header_style="bold magenta", **kw)
    for name, style in cols: t.add_column(name, style=style)
    return t

def welcome() -> None:
    from cpnlookup.llm import check_ollama
    console.print(f"[bold magenta]{BANNER}\n v{__version__} {CODENAME}: smarter retrieval, deeper context, persistent memory | by @projcjdevs[/]\n")
    ollama = "[bold green]Running[/]" if check_ollama() else "[bold red]Not found[/] [dim](needed for ask; run ollama serve)[/]"
    console.print(Panel(f"[bold cyan]Quick start[/]\n1. [green]lookup auth <token>[/]\n2. [green]lookup init <owner/repo>[/]\n3. [green]lookup ask \"How does X work?\"[/]\n\n"
                        f"[bold cyan]Ollama:[/] {ollama}\n[dim]lookup help for setup and all commands · lookup desc for project info[/]", border_style="magenta", box=box.ROUNDED))

def guide() -> None:
    console.print(Panel("1. Install Ollama from [bold cyan]ollama.com[/]\n2. Run [bold green]ollama serve[/]\n3. Pull a model: [bold green]ollama pull mistral[/]",
                        title="[bold white]Ollama setup[/]", border_style="blue", box=box.ROUNDED))
    console.print(Panel("1. GitHub [bold white]Settings > Developer settings[/]\n2. Create a [bold white]personal access token (classic)[/] with the [bold green]repo[/] scope\n"
                        "3. Run [bold green]lookup auth <token>[/]", title="[bold white]GitHub token[/]", border_style="green", box=box.ROUNDED))
    t = Table(box=box.SIMPLE, header_style="bold cyan"); t.add_column("Command", style="cyan"); t.add_column("What it does")
    for c, d in COMMANDS: t.add_row(f"lookup {esc(c)}", d)
    console.print(Panel(t, title="[bold white]Commands[/]", border_style="magenta"))

def desc() -> None:
    console.print(Panel(
        "[white]cpnlookup[/] indexes and queries GitHub repositories in natural language without [green]git clone[/].\n\n"
        "It combines [bold cyan]FAISS vector search[/], [bold cyan]BM25 keyword search[/] (SQLite FTS5), a [bold cyan]cross-encoder reranker[/]\n"
        "and a [bold cyan]static call graph[/] into a [bold magenta]hybrid RAG pipeline[/] that understands both the meaning and\n"
        "the structure of a codebase. Answers come from your local [bold]Ollama[/] model. No code, questions or\n"
        "embeddings leave your machine.", title="[bold white]What is cpnlookup?[/]", border_style="magenta", box=box.ROUNDED))
    log = Table(box=box.SIMPLE, header_style="bold cyan", show_edge=False)
    log.add_column("Version", style="bold magenta", no_wrap=True); log.add_column("Name", style="bold white", no_wrap=True); log.add_column("Changes", style="dim white")
    for row in [("v1.0.0", "Initial Release", "GitHub API indexing, AST chunking, FAISS search, SQLite call graph, Ollama answers."),
                ("v1.1.1", "Optimized Indexing", "Pre-index filtering, batched embedding, fixed runtime dependencies."),
                ("v1.2.2", "Lazy Imports", "Heavy ML imports moved into functions; light commands start instantly."),
                ("v2.0.0", "Forge", "Incremental indexing, write consistency, embedding storage, faiss_id mapping,\ncallers + callees in context."),
                ("v3.0.0", "Flux", "Hybrid retrieval (BM25 + FAISS + cross-encoder), conversation memory, history and\nforget. Leaner codebase, index schema v3, context budget, many correctness fixes.")]:
        log.add_row(*row)
    console.print(Panel(log, title="[bold white]Version history[/]", border_style="cyan", box=box.ROUNDED))
    console.print(Panel("[bold white]Developer[/]\n  [bold magenta]@projcjdevs[/] — architecture, implementation and maintenance\n\n[bold white]Special thanks[/]\n"
                        "  [bold cyan]@2nieGarcia[/]       — feedback and testing\n  [bold cyan]@renzv-compsci[/]    — feedback and testing\n"
                        "  [bold cyan]@NIghtIngale340[/]   — feedback and testing", title="[bold white]Acknowledgements[/]", border_style="green", box=box.ROUNDED))
