# rag_anywhere/cli/commands/documents.py
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional, List
import json

from ..context import RAGContext
from ...core.splitters import SplitterFactory

app = typer.Typer()
console = Console()


def _parse_splitter_overrides(
    splitter: Optional[str],
    ctx: typer.Context
) -> Optional[dict]:
    """
    Parse splitter overrides from CLI options.
    Handles arbitrary parameters for different splitter types.
    """
    if splitter is None:
        return None
    
    overrides = {'strategy': splitter}
    
    # Get all remaining options from context
    # This allows us to support arbitrary splitter parameters
    params = ctx.params
    
    # Common parameters for recursive splitter
    if 'chunk_size' in params and params['chunk_size'] is not None:
        overrides['chunk_size'] = params['chunk_size']
    if 'chunk_overlap' in params and params['chunk_overlap'] is not None:
        overrides['chunk_overlap'] = params['chunk_overlap']
    
    # Common parameters for structural splitter
    if 'min_chunk_size' in params and params['min_chunk_size'] is not None:
        overrides['min_chunk_size'] = params['min_chunk_size']
    if 'max_chunk_size' in params and params['max_chunk_size'] is not None:
        overrides['max_chunk_size'] = params['max_chunk_size']
    
    return overrides


@app.command()
def add(
    ctx: typer.Context,
    paths: List[Path] = typer.Argument(..., help="File(s) or directory to add", exists=True),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recursively add files from directories"),
    metadata: Optional[str] = typer.Option(None, "--metadata", "-m", help="JSON metadata to attach to document(s)"),
    splitter: Optional[str] = typer.Option(None, "--splitter", "-s", help="Splitter strategy (recursive, structural)"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", help="Chunk size in characters (for recursive splitter)"),
    chunk_overlap: Optional[int] = typer.Option(None, "--chunk-overlap", help="Chunk overlap in characters (for recursive splitter)"),
    min_chunk_size: Optional[int] = typer.Option(None, "--min-chunk-size", help="Minimum chunk size (for structural splitter)"),
    max_chunk_size: Optional[int] = typer.Option(None, "--max-chunk-size", help="Maximum chunk size (for structural splitter)"),
):
    """Add document(s) to the active database"""
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
    
    # Parse metadata
    parsed_metadata = None
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗[/red] Invalid JSON metadata: {e}", style="bold")
            raise typer.Exit(1)
    
    # Parse splitter overrides
    splitter_overrides = _parse_splitter_overrides(splitter, ctx)
    
    # Collect files to add
    files_to_add = []
    for path in paths:
        if path.is_file():
            files_to_add.append(path)
        elif path.is_dir():
            supported_exts = set(rag_ctx.loader_registry.get_supported_extensions())
            if recursive:
                files = [
                    f for f in path.rglob('*')
                    if f.is_file() and f.suffix.lower() in supported_exts
                ]
            else:
                files = [
                    f for f in path.glob('*')
                    if f.is_file() and f.suffix.lower() in supported_exts
                ]
            files_to_add.extend(files)
    
    if not files_to_add:
        console.print("[yellow]No supported files found[/yellow]")
        raise typer.Exit(0)
    
    console.print(f"Found {len(files_to_add)} file(s) to index")
    
    # Index files with progress
    success_count = 0
    error_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Indexing documents...", total=len(files_to_add))
        
        for file_path in files_to_add:
            try:
                # Get splitter config for this file type
                file_ext = file_path.suffix.lower()
                splitter_config = rag_ctx.get_splitter_config(file_ext, splitter_overrides)
                
                # Create indexer with appropriate splitter
                splitter = SplitterFactory.create_splitter(
                    splitter_config['strategy'],
                    token_estimator=rag_ctx.embedding_provider.estimate_tokens,
                    **{k: v for k, v in splitter_config.items() if k != 'strategy'}
                )
                
                # Temporarily replace the indexer's splitter
                original_splitter = rag_ctx.indexer.splitter
                rag_ctx.indexer.splitter = splitter
                
                # Index document
                doc_id = rag_ctx.indexer.index_document(file_path, parsed_metadata)
                
                # Restore original splitter
                rag_ctx.indexer.splitter = original_splitter
                
                console.print(f"[green]✓[/green] {file_path.name} (ID: {doc_id[:8]}...)")
                success_count += 1
                
            except ValueError as e:
                console.print(f"[yellow]⚠[/yellow] {file_path.name}: {e}")
                error_count += 1
            except Exception as e:
                console.print(f"[red]✗[/red] {file_path.name}: {e}")
                error_count += 1
            
            progress.advance(task)
    
    # Summary
    console.print()
    console.print(f"[bold]Summary:[/bold] {success_count} succeeded, {error_count} failed")


@app.command()
def remove(
    identifier: str = typer.Argument(..., help="Document ID or filename to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    by_id: bool = typer.Option(False, "--by-id", help="Treat identifier as document ID"),
    by_filename: bool = typer.Option(False, "--by-filename", help="Treat identifier as filename"),
):
    """Remove document(s) from the active database"""
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
    
    # Determine what we're searching for
    doc_to_remove = None
    
    if by_id or (not by_filename and len(identifier) == 36):  # UUID length
        # Try to find by ID
        doc_to_remove = rag_ctx.document_store.get_document(identifier)
        if not doc_to_remove:
            console.print(f"[red]✗[/red] Document with ID '{identifier}' not found", style="bold")
            raise typer.Exit(1)
    else:
        # Try to find by filename
        doc_to_remove = rag_ctx.document_store.get_document_by_filename(identifier)
        if not doc_to_remove:
            console.print(f"[red]✗[/red] Document with filename '{identifier}' not found", style="bold")
            raise typer.Exit(1)
    
    # Confirmation
    if not force:
        console.print(f"[yellow]⚠[/yellow]  Remove document: {doc_to_remove['filename']} (ID: {doc_to_remove['id']})?")
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Cancelled")
            raise typer.Exit(0)
    
    # Remove document
    try:
        success = rag_ctx.indexer.remove_document(doc_to_remove['id'])
        if success:
            console.print(f"[green]✓[/green] Removed document '{doc_to_remove['filename']}'", style="bold")
        else:
            console.print(f"[red]✗[/red] Failed to remove document", style="bold")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error removing document: {e}", style="bold")
        raise typer.Exit(1)


@app.command(name="list")
def list_documents(
    filter_json: Optional[str] = typer.Option(None, "--filter", "-f", help="JSON filter for metadata"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
):
    """List all documents in the active database"""
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
    
    # Parse filter
    filter_dict = None
    if filter_json:
        try:
            filter_dict = json.loads(filter_json)
        except json.JSONDecodeError as e:
            console.print(f"[red]✗[/red] Invalid JSON filter: {e}", style="bold")
            raise typer.Exit(1)
    
    # Get documents
    documents = rag_ctx.document_store.list_documents()
    
    if not documents:
        console.print("No documents in database")
        return
    
    # Apply filter if provided
    if filter_dict:
        filtered_docs = []
        for doc in documents:
            doc_metadata = doc.get('metadata', {})
            # Check if all filter key-value pairs match
            if all(doc_metadata.get(k) == v for k, v in filter_dict.items()):
                filtered_docs.append(doc)
        documents = filtered_docs
    
    if not documents:
        console.print("No documents match the filter")
        return
    
    # Display documents
    if verbose:
        for doc in documents:
            console.print(f"\n[bold cyan]{doc['filename']}[/bold cyan]")
            console.print(f"  ID: {doc['id']}")
            console.print(f"  Created: {doc['created_at']}")
            if doc['metadata']:
                console.print(f"  Metadata: {json.dumps(doc['metadata'])}")
            
            # Get chunk count
            chunks = rag_ctx.document_store.get_chunks_by_document(doc['id'])
            console.print(f"  Chunks: {len(chunks)}")
    else:
        table = Table(title=f"Documents in '{db_name}'")
        table.add_column("Filename", style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Created", style="magenta")
        table.add_column("Chunks", style="green")
        
        for doc in documents:
            chunks = rag_ctx.document_store.get_chunks_by_document(doc['id'])
            table.add_row(
                doc['filename'],
                doc['id'][:8] + "...",
                doc['created_at'][:10],  # Just the date
                str(len(chunks))
            )
        
        console.print(table)
        console.print(f"\nTotal: {len(documents)} document(s)")
