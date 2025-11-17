# rag_anywhere/cli/commands/search.py
import typer
from rich.console import Console
from rich.panel import Panel
from typing import Optional
import requests

from ..context import RAGContext
from ...server.manager import ServerManager

app = typer.Typer()
console = Console()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    min_score: Optional[float] = typer.Option(None, "--min-score", "-s", help="Minimum similarity score (0-1)"),
    context: int = typer.Option(0, "--context", "-c", help="Number of surrounding chunks to include"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """Search for documents in the active database"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
    try:
        ctx.ensure_active_database()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    # Ensure server is running
    try:
        manager.ensure_server_running()
    except Exception as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    # Get server info
    status = manager.get_status()
    port = status['port']
    
    # Perform search via API
    console.print(f"Searching for: [bold]{query}[/bold]\n")
    
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/search",
            json={
                'query': query,
                'top_k': top_k,
                'min_score': min_score
            },
            timeout=30
        )
        
        if response.status_code != 200:
            console.print(f"[red]✗[/red] Search failed: {response.text}", style="bold")
            raise typer.Exit(1)
        
        data = response.json()
        results = data['results']
        
    except requests.RequestException as e:
        console.print(f"[red]✗[/red] Failed to communicate with server: {e}", style="bold")
        console.print("\nTry restarting the server: rag-anywhere server restart")
        raise typer.Exit(1)
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    # Display results
    console.print(f"Found {len(results)} result(s):\n")
    
    for i, result in enumerate(results, 1):
        # Header with score and document info
        header = (
            f"[bold cyan]Result {i}[/bold cyan] "
            f"[dim](Score: {result['similarity_score']:.3f})[/dim]\n"
            f"[bold]Document:[/bold] {result['document']['filename']}\n"
            f"[bold]Position:[/bold] Chunk {result['position']['chunk_index']} "
            f"(chars {result['position']['start_char']}-{result['position']['end_char']})"
        )
        
        content = result['content']
        
        # TODO: Implement context fetching via API if needed
        if context > 0:
            header += f"\n[dim](Context fetching not yet implemented via API)[/dim]"
        
        # Create panel with content
        panel = Panel(
            content,
            title=header,
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)
        
        # Show metadata if requested
        if show_metadata and result.get('metadata'):
            console.print(f"[dim]Metadata: {result['metadata']}[/dim]")
        
        # Show document ID for reference
        console.print(f"[dim]Document ID: {result['document']['id']}[/dim]")
        console.print()
