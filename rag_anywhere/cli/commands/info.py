# rag_anywhere/cli/commands/info.py

import typer
from rich.console import Console
from rich.table import Table

from ..context import RAGContext
from ...server.manager import ServerManager
from ...server.state import ServerStatus

app = typer.Typer()
console = Console()


@app.command()
def show_status():
    """Show overall system status"""
    rag_ctx = RAGContext()

    databases = rag_ctx.config.list_databases()
    active_db = rag_ctx.config.get_active_database()

    # Get server status
    server_manager = ServerManager(rag_ctx.config)
    server_status = server_manager.get_status()

    console.print("\n[bold]RAG Anywhere - System Status[/bold]")
    console.print(f"[dim]{'─' * 60}[/dim]")

    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Config directory: {rag_ctx.config.config_dir}")
    console.print(f"  Databases directory: {rag_ctx.config.databases_dir}")

    # Server status section
    console.print(f"\n[bold]Server:[/bold]")
    status_value = server_status.get('status', ServerStatus.STOPPED.value)

    if status_value == ServerStatus.RUNNING.value:
        console.print(f"  Status: [green]RUNNING[/green]")
    elif status_value == ServerStatus.SLEEPING.value:
        console.print(f"  Status: [yellow]SLEEPING[/yellow]")
    elif status_value == ServerStatus.STOPPED.value:
        console.print(f"  Status: [dim]STOPPED[/dim]")
    else:
        console.print(f"  Status: [red]{status_value}[/red]")

    console.print(f"\n[bold]Databases:[/bold]")
    if not databases:
        console.print("  No databases found")
    else:
        console.print(f"  Total: {len(databases)}")
        console.print(f"  Active: {active_db or 'None'}")

        # Show table of databases
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Name")
        table.add_column("Provider")
        table.add_column("Status")

        for db_name in sorted(databases):
            try:
                from ...config.embedding_config import EMBEDDING_MODEL
                # Embedding model is now global
                provider = EMBEDDING_MODEL
                status = "[green]ACTIVE[/green]" if db_name == active_db else ""
                table.add_row(db_name, provider, status)
            except Exception:
                table.add_row(db_name, "[red]ERROR[/red]", "")

        console.print()
        console.print(table)

    console.print()
