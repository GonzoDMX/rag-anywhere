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


def _perform_semantic_search(
    query: str,
    task: str,
    top_k: int,
    min_score: Optional[float],
    context: int,
    show_metadata: bool,
    search_label: str
):
    """Common function for all semantic search variants"""
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
    console.print(f"{search_label}: [bold]{query}[/bold]\n")

    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/search",
            json={
                'query': query,
                'top_k': top_k,
                'min_score': min_score,
                'task': task
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


@app.command(name="semantic")
def semantic_search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    min_score: Optional[float] = typer.Option(None, "--min-score", "-s", help="Minimum similarity score (0-1)"),
    context: int = typer.Option(0, "--context", "-c", help="Number of surrounding chunks to include"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """Semantic search for general information retrieval (default search mode)"""
    _perform_semantic_search(
        query=query,
        task="retrieval",
        top_k=top_k,
        min_score=min_score,
        context=context,
        show_metadata=show_metadata,
        search_label="Semantic search"
    )


@app.command(name="code")
def code_search(
    query: str = typer.Argument(..., help="Natural language query describing code to find"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    min_score: Optional[float] = typer.Option(None, "--min-score", "-s", help="Minimum similarity score (0-1)"),
    context: int = typer.Option(0, "--context", "-c", help="Number of surrounding chunks to include"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """Search for code using natural language queries (optimized for code retrieval)"""
    _perform_semantic_search(
        query=query,
        task="code_retrieval",
        top_k=top_k,
        min_score=min_score,
        context=context,
        show_metadata=show_metadata,
        search_label="Code search"
    )


@app.command(name="question")
def question_search(
    query: str = typer.Argument(..., help="Question to answer"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    min_score: Optional[float] = typer.Option(None, "--min-score", "-s", help="Minimum similarity score (0-1)"),
    context: int = typer.Option(0, "--context", "-c", help="Number of surrounding chunks to include"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """Search for answers to questions (optimized for question answering)"""
    _perform_semantic_search(
        query=query,
        task="question_answering",
        top_k=top_k,
        min_score=min_score,
        context=context,
        show_metadata=show_metadata,
        search_label="Question answering search"
    )


@app.command(name="facts")
def fact_check_search(
    query: str = typer.Argument(..., help="Statement to verify"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
    min_score: Optional[float] = typer.Option(None, "--min-score", "-s", help="Minimum similarity score (0-1)"),
    context: int = typer.Option(0, "--context", "-c", help="Number of surrounding chunks to include"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """Search for evidence to verify facts (optimized for fact checking)"""
    _perform_semantic_search(
        query=query,
        task="fact_checking",
        top_k=top_k,
        min_score=min_score,
        context=context,
        show_metadata=show_metadata,
        search_label="Fact checking search"
    )


@app.command(name="keyword")
def keyword(
    query: Optional[str] = typer.Argument(None, help="Keyword search query (supports AND, OR, NOT, phrases, prefix)"),
    required: Optional[str] = typer.Option(None, "--required", "-r", help="Comma-separated required keywords (all must be present)"),
    optional: Optional[str] = typer.Option(None, "--optional", "-o", help="Comma-separated optional keywords (at least one)"),
    exclude: Optional[str] = typer.Option(None, "--exclude", "-e", help="Comma-separated terms/keywords to exclude"),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return"),
    highlight: bool = typer.Option(True, "--highlight/--no-highlight", help="Highlight matched terms"),
    exact_match: bool = typer.Option(False, "--exact-match", help="Treat query as exact phrase match (free-form mode only)"),
    show_metadata: bool = typer.Option(False, "--metadata", "-m", help="Show chunk metadata"),
):
    """
    Keyword search using FTS5 full-text search (no AI embeddings).

    Two modes available:

    **Free-form mode** (provide query argument):
      - Simple queries: "machine learning"
      - Boolean operators: "machine AND learning", "machine OR learning"
      - NOT operator: "machine NOT cat"
      - Phrase queries: '"machine learning"' (exact phrase)
      - Prefix matching: "mach*"

    **Structured mode** (provide --required and/or --optional):
      - Required terms (AND): --required "machine,learning"
      - Optional terms (OR): --optional "neural,deep"
      - Exclude terms: --exclude "cat,bird"

    Examples:
      # Free-form mode
      rag-anywhere search keyword "machine learning"
      rag-anywhere search keyword "machine AND learning"
      rag-anywhere search keyword "machine" --exact-match
      rag-anywhere search keyword "dog" --exclude "cat,bird"

      # Structured mode
      rag-anywhere search keyword --required "machine,learning"
      rag-anywhere search keyword --required "machine" --optional "neural,deep"
      rag-anywhere search keyword --required "dog" --exclude "cat,bird"
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

    # Detect mode and validate
    has_freeform = query is not None
    has_structured = required is not None or optional is not None

    if not has_freeform and not has_structured:
        console.print("[red]✗[/red] Must provide either a query argument or --required/--optional flags", style="bold")
        console.print("\nExamples:")
        console.print('  rag-anywhere search keyword "machine learning"')
        console.print('  rag-anywhere search keyword --required "machine,learning"')
        raise typer.Exit(1)

    if has_freeform and has_structured:
        console.print("[red]✗[/red] Cannot mix free-form query with --required/--optional flags", style="bold")
        console.print("\nUse either:")
        console.print('  - Free-form: rag-anywhere search keyword "machine AND learning"')
        console.print('  - Structured: rag-anywhere search keyword --required "machine,learning"')
        raise typer.Exit(1)

    # Build request based on mode
    if has_freeform:
        # Free-form mode
        exclude_terms = [term.strip() for term in exclude.split(',')] if exclude else None

        console.print(f"Keyword search for: [bold]{query}[/bold]")
        if exclude_terms:
            console.print(f"[dim]Excluding: {', '.join(exclude_terms)}[/dim]")
        console.print()

        request_body = {
            'query': query,
            'top_k': top_k,
            'exclude_terms': exclude_terms,
            'highlight': highlight,
            'exact_match': exact_match
        }
    else:
        # Structured mode
        required_keywords = [kw.strip() for kw in required.split(',')] if required else None
        optional_keywords = [kw.strip() for kw in optional.split(',')] if optional else None
        exclude_keywords = [kw.strip() for kw in exclude.split(',')] if exclude else None

        if exact_match:
            console.print("[yellow]⚠[/yellow] --exact-match is only supported in free-form mode (ignored)", style="bold")

        console.print(f"[bold]Keyword search (structured mode):[/bold]")
        if required_keywords:
            console.print(f"  Required (AND): {', '.join(required_keywords)}")
        if optional_keywords:
            console.print(f"  Optional (OR): {', '.join(optional_keywords)}")
        if exclude_keywords:
            console.print(f"  Exclude (NOT): {', '.join(exclude_keywords)}")
        console.print()

        request_body = {
            'required_keywords': required_keywords,
            'optional_keywords': optional_keywords,
            'exclude_keywords': exclude_keywords,
            'top_k': top_k,
            'highlight': highlight
        }

    # Perform keyword search via API
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/search/keyword",
            json=request_body,
            timeout=30
        )

        if response.status_code == 400:
            error = response.json().get('detail', 'Invalid query')
            console.print(f"[red]✗[/red] {error}", style="bold")
            if has_freeform:
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
