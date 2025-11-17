# rag_anywhere/cli/commands/server.py
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

from ..context import RAGContext
from ...server.manager import ServerManager
from ...server.state import ServerStatus
from ...utils import get_logger

app = typer.Typer()
console = Console()
logger = get_logger('cli.server')


@app.command()
def start(
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to run server on"),
    force: bool = typer.Option(False, "--force", "-f", help="Force restart if already running"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Run in debug mode with verbose output")
):
    """Start the RAG Anywhere server"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
    # Check if there's an active database
    active_db = ctx.config.get_active_database()
    if not active_db:
        console.print(
            "[red]✗[/red] No active database. Create or activate a database first:",
            style="bold"
        )
        console.print("  rag-anywhere db create <name>")
        console.print("  rag-anywhere db use <name>")
        raise typer.Exit(1)
    
    # Check current status
    status = manager.get_status()
    
    if status['status'] == ServerStatus.RUNNING.value and not force:
        console.print(f"[yellow]⚠[/yellow]  Server already running on port {status['port']}")
        console.print(f"  Database: {status['active_db']}")
        console.print(f"  PID: {status['pid']}")
        console.print("\nUse --force to restart")
        return
    
    try:
        console.print(f"Starting server for database '{active_db}'...")
        if debug:
            console.print("[yellow]Debug mode enabled - server output will be visible[/yellow]")
        
        manager.start_server(port=port, force=force, debug=debug)
        
        status = manager.get_status()
        console.print(f"[green]✓[/green] Server started successfully", style="bold")
        console.print(f"  Port: {status['port']}")
        console.print(f"  Database: {status['active_db']}")
        console.print(f"  PID: {status['pid']}")
        console.print(f"  Embedding model: {status['embedding_model']}")
        
        # Show log location
        log_dir = ctx.config.config_dir / "logs"
        console.print(f"\n[dim]Logs: {log_dir}/server-*.log[/dim]")
        
    except RuntimeError as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        
        # Show log location for troubleshooting
        log_dir = ctx.config.config_dir / "logs"
        console.print(f"\n[yellow]Check logs for details:[/yellow]")
        console.print(f"  {log_dir}/server-stderr.log")
        console.print(f"  {log_dir}/rag-anywhere.log")
        
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to start server: {e}", style="bold")
        raise typer.Exit(1)


@app.command()
def stop():
    """Stop the RAG Anywhere server"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
    status = manager.get_status()
    
    if status['status'] == ServerStatus.STOPPED.value:
        console.print("[yellow]Server is not running[/yellow]")
        return
    
    console.print("Stopping server...")
    success = manager.stop_server()
    
    if success:
        console.print("[green]✓[/green] Server stopped", style="bold")
    else:
        console.print("[red]✗[/red] Failed to stop server", style="bold")
        raise typer.Exit(1)


@app.command()
def restart(
    port: Optional[int] = typer.Option(None, "--port", "-p", help="Port to run server on"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Run in debug mode with verbose output")
):
    """Restart the RAG Anywhere server"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
    console.print("Restarting server...")
    
    try:
        manager.restart_server(port=port, debug=debug)
        
        status = manager.get_status()
        console.print(f"[green]✓[/green] Server restarted successfully", style="bold")
        console.print(f"  Port: {status['port']}")
        console.print(f"  Database: {status['active_db']}")
        console.print(f"  PID: {status['pid']}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to restart server: {e}", style="bold")
        
        # Show log location
        log_dir = ctx.config.config_dir / "logs"
        console.print(f"\n[yellow]Check logs for details:[/yellow]")
        console.print(f"  {log_dir}/server-stderr.log")
        
        raise typer.Exit(1)


@app.command()
def status():
    """Show server status"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
    status = manager.get_status()
    
    console.print("\n[bold]RAG Anywhere Server Status[/bold]")
    console.print(f"[dim]{'─' * 60}[/dim]")
    
    # Status with color
    status_str = status['status']
    if status_str == ServerStatus.RUNNING.value:
        status_display = f"[green]{status_str.upper()}[/green]"
    elif status_str == ServerStatus.SLEEPING.value:
        status_display = f"[yellow]{status_str.upper()}[/yellow]"
    elif status_str == ServerStatus.CRASHED.value:
        status_display = f"[red]{status_str.upper()}[/red]"
    else:
        status_display = f"[dim]{status_str.upper()}[/dim]"
    
    console.print(f"\nStatus: {status_display}")
    
    if status['status'] != ServerStatus.STOPPED.value:
        console.print(f"PID: {status['pid']}")
        console.print(f"Port: {status['port']}")
        console.print(f"Database: {status['active_db']}")
        console.print(f"Embedding Model: {status['embedding_model']}")
        
        if status['last_activity']:
            console.print(f"Last Activity: {status['last_activity']}")
    
    if status['status'] == ServerStatus.CRASHED.value:
        console.print("\n[red]Server has crashed. Restart with:[/red]")
        console.print("  rag-anywhere server restart")
        
        # Show logs
        log_dir = ctx.config.config_dir / "logs"
        console.print(f"\n[yellow]Check logs:[/yellow]")
        console.print(f"  {log_dir}/server-stderr.log")
    
    console.print()


@app.command()
def logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log output"),
    stderr: bool = typer.Option(False, "--stderr", help="Show stderr instead of stdout")
):
    """Show server logs"""
    ctx = RAGContext()
    log_dir = ctx.config.config_dir / "logs"
    
    log_file = log_dir / ("server-stderr.log" if stderr else "server-stdout.log")
    
    if not log_file.exists():
        console.print(f"[yellow]Log file not found: {log_file}[/yellow]")
        return
    
    if follow:
        # Tail -f equivalent
        import subprocess
        subprocess.run(["tail", "-f", str(log_file)])
    else:
        # Show last 50 lines
        with open(log_file, 'r') as f:
            lines = f.readlines()
            for line in lines[-50:]:
                print(line, end='')
