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


@app.command(name="semantic")
def semantic_search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    min_score: Optional[float] = typer.Option(None, "--min-score", "-s", help="Minimum similarity score (0-1)"),
    context: int = typer.Option(0, "--context", "-c", help="Number of surrounding chunks to include"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """Semantic search using dense embeddings (default)"""
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
            f"[dim cyan](Score: {result['similarity_score']:.3f})[/dim cyan]\n"
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


@app.command(name="keyword")
def keyword(
    query: str = typer.Argument(..., help="Keyword search query (supports AND, OR, NOT, phrases, prefix)"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Comma-separated terms to exclude"),
    highlight: bool = typer.Option(True, "--highlight/--no-highlight", help="Highlight matched terms"),
    exact_match: bool = typer.Option(False, "--exact-match", help="Treat query as exact phrase match"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """
    Search using keywords with FTS5 full-text search.

    Examples:
      rag-anywhere search keyword "machine learning"
      rag-anywhere search keyword "machine AND learning"
      rag-anywhere search keyword "machine" --exact-match
      rag-anywhere search keyword "mach*" --highlight
      rag-anywhere search keyword "dog" --exclude "cat,bird"
    """
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

    # Parse exclude terms
    exclude_terms = None
    if exclude:
        exclude_terms = [term.strip() for term in exclude.split(',')]

    # Perform keyword search via API
    console.print(f"Keyword search for: [bold]{query}[/bold]")
    if exclude_terms:
        console.print(f"[dim]Excluding: {', '.join(exclude_terms)}[/dim]")
    console.print()

    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/search/keyword",
            json={
                'query': query,
                'top_k': top_k,
                'exclude_terms': exclude_terms,
                'highlight': highlight,
                'exact_match': exact_match
            },
            timeout=30
        )

        if response.status_code == 400:
            error = response.json().get('detail', 'Invalid query syntax')
            console.print(f"[red]✗[/red] {error}", style="bold")
            console.print("\n[dim]FTS5 Query Syntax Help:[/dim]")
            console.print("  - Simple: machine learning")
            console.print("  - AND: machine AND learning")
            console.print("  - OR: machine OR learning")
            console.print("  - NOT: machine NOT cat")
            console.print('  - Phrase: "machine learning"')
            console.print("  - Prefix: mach*")
            raise typer.Exit(1)

        if response.status_code != 200:
            console.print(f"[red]✗[/red] Search failed: {response.text}", style="bold")
            raise typer.Exit(1)

        data = response.json()
        results = data['results']

    except requests.RequestException as e:
        console.print(f"[red]✗[/red] Failed to communicate with server: {e}", style="bold")
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
            f"[dim cyan](Score: {result['score']:.3f})[/dim cyan]\n"
            f"[bold]Document:[/bold] {result['document']['filename']}\n"
            f"[bold]Position:[/bold] Chunk {result['position']['chunk_index']} "
            f"(chars {result['position']['start_char']}-{result['position']['end_char']})"
        )

        content = result['content']

        # Create panel with content
        panel = Panel(
            content,
            title=header,
            border_style="green",
            padding=(1, 2)
        )
        console.print(panel)

        # Show metadata if requested
        if show_metadata and result.get('metadata'):
            console.print(f"[dim]Metadata: {result['metadata']}[/dim]")

        # Show document ID for reference
        console.print(f"[dim]Document ID: {result['document']['id']}[/dim]")
        console.print()


@app.command(name="keyword-advanced")
def keyword_advanced(
    required: str = typer.Argument(..., help="Comma-separated required keywords (all must be present)"),
    optional: Optional[str] = typer.Option(None, "--optional", "-o", help="Comma-separated optional keywords (at least one)"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Comma-separated keywords to exclude"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return"),
    highlight: bool = typer.Option(True, "--highlight/--no-highlight", help="Highlight matched terms"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """
    Advanced keyword search with explicit control over required/optional/excluded terms.

    Examples:
      rag-anywhere search keyword-advanced "machine,learning"
      rag-anywhere search keyword-advanced "machine" --optional "neural,deep"
      rag-anywhere search keyword-advanced "dog" --exclude "cat,bird"
    """
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

    # Parse keywords
    required_keywords = [kw.strip() for kw in required.split(',')]
    optional_keywords = [kw.strip() for kw in optional.split(',')] if optional else None
    exclude_keywords = [kw.strip() for kw in exclude.split(',')] if exclude else None

    # Display query
    console.print(f"[bold]Advanced keyword search:[/bold]")
    console.print(f"  Required (AND): {', '.join(required_keywords)}")
    if optional_keywords:
        console.print(f"  Optional (OR): {', '.join(optional_keywords)}")
    if exclude_keywords:
        console.print(f"  Exclude (NOT): {', '.join(exclude_keywords)}")
    console.print()

    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/search/keyword/advanced",
            json={
                'required_keywords': required_keywords,
                'optional_keywords': optional_keywords,
                'exclude_keywords': exclude_keywords,
                'top_k': top_k,
                'highlight': highlight
            },
            timeout=30
        )

        if response.status_code == 400:
            error = response.json().get('detail', 'Invalid query')
            console.print(f"[red]✗[/red] {error}", style="bold")
            raise typer.Exit(1)

        if response.status_code != 200:
            console.print(f"[red]✗[/red] Search failed: {response.text}", style="bold")
            raise typer.Exit(1)

        data = response.json()
        results = data['results']

    except requests.RequestException as e:
        console.print(f"[red]✗[/red] Failed to communicate with server: {e}", style="bold")
        raise typer.Exit(1)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    # Display results
    console.print(f"Found {len(results)} result(s):\n")

    for i, result in enumerate(results, 1):
        header = (
            f"[bold cyan]Result {i}[/bold cyan] "
            f"[dim cyan](Score: {result['score']:.3f})[/dim cyan]\n"
            f"[bold]Document:[/bold] {result['document']['filename']}\n"
            f"[bold]Position:[/bold] Chunk {result['position']['chunk_index']} "
            f"(chars {result['position']['start_char']}-{result['position']['end_char']})"
        )

        content = result['content']

        panel = Panel(
            content,
            title=header,
            border_style="green",
            padding=(1, 2)
        )
        console.print(panel)

        if show_metadata and result.get('metadata'):
            console.print(f"[dim]Metadata: {result['metadata']}[/dim]")

        console.print(f"[dim]Document ID: {result['document']['id']}[/dim]")
        console.print()
