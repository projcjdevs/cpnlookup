import click
from rich.console import Console
from rich.table import Table
from cpnlookup.github.client import get_user_repos
from cpnlookup.utils.config import save_github_token

console = Console()

@click.group()
def cli():
    """
    \b
     ██████╗██████╗ ███╗   ██╗██╗      ██████╗  ██████╗ ██╗  ██╗██╗   ██╗██████╗ 
    ██╔════╝██╔══██╗████╗  ██║██║     ██╔═══██╗██╔═══██╗██║ ██╔╝██║   ██║██╔══██╗
    ██║     ██████╔╝██╔██╗ ██║██║     ██║   ██║██║   ██║█████╔╝ ██║   ██║██████╔╝
    ██║     ██╔═══╝ ██║╚██╗██║██║     ██║   ██║██║   ██║██╔═██╗ ██║   ██║██╔═══╝ 
    ╚██████╗██║     ██║ ╚████║███████╗╚██████╔╝╚██████╔╝██║  ██╗╚██████╔╝██║     
     ╚═════╝╚═╝     ╚═╝  ╚═══╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝     

    Local CLI tool for GitHub RAG, Call Graphs, and Codebase querying.
    """
    pass

# Auth Command
@cli.command()
@click.argument('token')
def auth(token: str):
    """Save your GitHub Personal Access Token."""
    save_github_token(token)
    
    console.print("[bold green]✓[/bold green] GitHub token saved successfully to ~/.cpnlookup/auth.json")

# Profile Command
@cli.command()
@click.argument('username')
def profile(username: str):
    """List the most recently updated repositories for a GitHub user."""
    
    with console.status(f"[bold cyan]Fetching repos for {username}..."):
        try:
            repos = get_user_repos(username)
        except Exception as e:
            console.print(f"[bold red]Error fetching repos:[/] {e}")
            return
            
    if not repos:
        console.print(f"[yellow]No repositories found for {username}.[/]")
        return

    table = Table(title=f"{username}'s Repositories", show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=4)
    table.add_column("Repo Name", style="cyan")
    table.add_column("Stars", justify="right", style="yellow")
    table.add_column("Language", style="green")
    
    for i, repo in enumerate(repos[:20], 1):
        table.add_row(
            str(i),
            repo.get("name", "Unknown"),
            str(repo.get("stargazers_count", 0)),
            repo.get("language") or "N/A"
        )
        
    console.print(table)