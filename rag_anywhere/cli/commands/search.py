# rag_anywhere/cli/commands/search.py
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from typing import Optional

from ..context import RAGContext

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
    rag_ctx = RAGContext()
    
    try:
        rag_ctx.ensure_active_database()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    # Load active database
    db_name = rag_ctx.get_active_database_name()
    try:
        rag_ctx.load_database(db_name, verbose=False)
    except Exception as e:
        console.print(f"[red]✗[/red] Error loading database: {e}", style="bold")
        raise typer.Exit(1)
    
    # Perform search
    console.print(f"Searching for: [bold]{query}[/bold]\n")
    
    try:
        results = rag_ctx.searcher.search(query, top_k=top_k, min_score=min_score)
    except Exception as e:
        console.print(f"[red]✗[/red] Search error: {e}", style="bold")
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
            f"[dim](Score: {result.similarity_score:.3f})[/dim]\n"
            f"[bold]Document:[/bold] {result.document_filename}\n"
            f"[bold]Position:[/bold] Chunk {result.chunk_index} "
            f"(chars {result.start_char}-{result.end_char})"
        )
        
        # Get content (with context if requested)
        if context > 0:
            content = rag_ctx.searcher.get_document_context(
                result.document_id,
                result.chunk_index,
                context_chunks=context
            )
            header += f"\n[dim](Showing {context} chunks before/after for context)[/dim]"
        else:
            content = result.chunk_content
        
        # Create panel with content
        panel = Panel(
            content,
            title=header,
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)
        
        # Show metadata if requested
        if show_metadata and result.metadata:
            console.print(f"[dim]Metadata: {result.metadata}[/dim]")
        
        # Show document ID for reference
        console.print(f"[dim]Document ID: {result.document_id}[/dim]")
        console.print()
