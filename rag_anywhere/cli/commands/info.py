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
def show_info():
    """Show information about the active database"""
    rag_ctx = RAGContext()
    
    try:
        rag_ctx.ensure_active_database()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    db_name = rag_ctx.get_active_database_name()
    
    # This will call the db info command
    from .db import info as db_info_cmd
    db_info_cmd(db_name)


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
                config = rag_ctx.config.load_database_config(db_name)
                provider = config['embedding']['provider']
                status = "[green]ACTIVE[/green]" if db_name == active_db else ""
                table.add_row(db_name, provider, status)
            except Exception:
                table.add_row(db_name, "[red]ERROR[/red]", "")

        console.print()
        console.print(table)

    console.print()
