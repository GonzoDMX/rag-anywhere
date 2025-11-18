# rag_anywhere/cli/commands/documents.py
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
import requests

from ..context import RAGContext
from ...server.manager import ServerManager

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
    manager = ServerManager(rag_ctx.config)
    
    try:
        rag_ctx.ensure_active_database()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    # Ensure server is running
    try:
        manager.ensure_server_running()
    except Exception as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        console.print("\nTry starting the server: rag-anywhere server start")
        raise typer.Exit(1)
    
    # Get server port
    status = manager.get_status()
    port = status['port']
    
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
    # Get supported extensions from registry (need to create one temporarily)
    from ...core.loaders import LoaderRegistry
    temp_registry = LoaderRegistry()
    supported_exts = set(temp_registry.get_supported_extensions())
    
    files_to_add = []
    for path in paths:
        if path.is_file():
            files_to_add.append(path)
        elif path.is_dir():
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

    # Build batch request
    documents_batch = []
    for file_path in files_to_add:
        documents_batch.append({
            'file_path': str(file_path.absolute()),
            'metadata': parsed_metadata,
            'splitter_overrides': splitter_overrides
        })

    # Make batch API request
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Indexing documents...", total=len(files_to_add))

            response = requests.post(
                f"http://127.0.0.1:{port}/documents/add-batch",
                json={
                    'documents': documents_batch,
                    'fail_fast': False
                },
                timeout=600  # 10 minutes for batch
            )

            progress.update(task, completed=len(files_to_add))

        if response.status_code == 200:
            data = response.json()

            # Display results
            for result in data['results']:
                if result['status'] == 'success':
                    doc_id = result['document_id']
                    filename = result['filename']
                    console.print(f"[green]✓[/green] {filename} (ID: {doc_id[:8]}...)")
                else:
                    file_path = Path(result['file_path'])
                    error_msg = result['error']
                    console.print(f"[red]✗[/red] {file_path.name}: {error_msg}")

            # Summary
            console.print()
            summary = data['summary']
            console.print(f"[bold]Summary:[/bold] {summary['succeeded']} succeeded, {summary['failed']} failed")
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            error_msg = error_data.get('detail', response.text)
            console.print(f"[red]✗[/red] Batch request failed: {error_msg}")
            raise typer.Exit(1)

    except requests.Timeout:
        console.print(f"[red]✗[/red] Batch request timed out")
        raise typer.Exit(1)
    except requests.RequestException as e:
        console.print(f"[red]✗[/red] Failed to communicate with server: {e}")
        raise typer.Exit(1)


@app.command()
def remove(
    identifier: str = typer.Argument(..., help="Document ID or filename to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
    by_id: bool = typer.Option(False, "--by-id", help="Treat identifier as document ID"),
    by_filename: bool = typer.Option(False, "--by-filename", help="Treat identifier as filename"),
):
    """Remove document(s) from the active database"""
    rag_ctx = RAGContext()
    manager = ServerManager(rag_ctx.config)
    
    try:
        rag_ctx.ensure_active_database()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    # Ensure server is running
    try:
        manager.ensure_server_running()
    except Exception as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        console.print("\nTry starting the server: rag-anywhere server start")
        raise typer.Exit(1)
    
    # Get server port
    status = manager.get_status()
    port = status['port']
    
    # First, we need to resolve the identifier to a document ID
    doc_to_remove = None
    
    try:
        if by_id or (not by_filename and len(identifier) == 36):  # UUID length
            # Try to get by ID directly
            response = requests.get(
                f"http://127.0.0.1:{port}/documents/{identifier}",
                timeout=10
            )
            
            if response.status_code == 200:
                doc_to_remove = response.json()
            else:
                console.print(f"[red]✗[/red] Document with ID '{identifier}' not found", style="bold")
                raise typer.Exit(1)
        else:
            # Search by filename
            response = requests.get(
                f"http://127.0.0.1:{port}/documents/list",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                documents = data['documents']
                
                # Find document with matching filename
                matches = [doc for doc in documents if doc['filename'] == identifier]
                
                if not matches:
                    console.print(f"[red]✗[/red] Document with filename '{identifier}' not found", style="bold")
                    raise typer.Exit(1)
                elif len(matches) > 1:
                    console.print(f"[yellow]⚠[/yellow]  Multiple documents found with filename '{identifier}':")
                    for doc in matches:
                        console.print(f"  - ID: {doc['id']}")
                    console.print("\nUse --by-id to specify which one to remove")
                    raise typer.Exit(1)
                else:
                    doc_to_remove = matches[0]
            else:
                console.print("[red]✗[/red] Failed to list documents", style="bold")
                raise typer.Exit(1)
    
    except requests.RequestException as e:
        console.print(f"[red]✗[/red] Failed to communicate with server: {e}", style="bold")
        raise typer.Exit(1)
    
    # Confirmation
    if not force:
        console.print(f"[yellow]⚠[/yellow]  Remove document: {doc_to_remove['filename']} (ID: {doc_to_remove['id']})?")
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Cancelled")
            raise typer.Exit(0)
    
    # Remove document via API
    try:
        response = requests.post(
            f"http://127.0.0.1:{port}/documents/remove",
            json={'document_id': doc_to_remove['id']},
            timeout=30
        )
        
        if response.status_code == 200:
            console.print(f"[green]✓[/green] Removed document '{doc_to_remove['filename']}'", style="bold")
        else:
            error_data = response.json() if response.headers.get('content-type') == 'application/json' else {}
            error_msg = error_data.get('detail', response.text)
            console.print(f"[red]✗[/red] Failed to remove document: {error_msg}", style="bold")
            raise typer.Exit(1)
    
    except requests.RequestException as e:
        console.print(f"[red]✗[/red] Failed to communicate with server: {e}", style="bold")
        raise typer.Exit(1)


@app.command(name="list")
def list_documents(
    filter_json: Optional[str] = typer.Option(None, "--filter", "-f", help="Filter by metadata keys (comma-separated) or key-value pairs (JSON)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information"),
    sort_by: str = typer.Option("created", "--sort", "-s", help="Sort by: created, filename, chunks"),
    order: str = typer.Option("desc", "--order", "-o", help="Sort order: asc or desc"),
    page: int = typer.Option(1, "--page", "-p", help="Page number (starts at 1)", min=1),
    per_page: int = typer.Option(20, "--per-page", help="Results per page", min=1, max=100),
):
    """List all documents in the active database"""
    rag_ctx = RAGContext()
    manager = ServerManager(rag_ctx.config)
    
    try:
        rag_ctx.ensure_active_database()
    except ValueError as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        raise typer.Exit(1)
    
    # Ensure server is running
    try:
        manager.ensure_server_running()
    except Exception as e:
        console.print(f"[red]✗[/red] {e}", style="bold")
        console.print("\nTry starting the server: rag-anywhere server start")
        raise typer.Exit(1)
    
    # Get server port
    status = manager.get_status()
    port = status['port']
    db_name = status['active_db']

    # Validate sort_by and order
    valid_sort_by = ['created', 'filename', 'chunks']
    if sort_by not in valid_sort_by:
        console.print(f"[red]✗[/red] Invalid sort_by value. Must be one of: {', '.join(valid_sort_by)}", style="bold")
        raise typer.Exit(1)

    if order not in ['asc', 'desc']:
        console.print(f"[red]✗[/red] Invalid order value. Must be 'asc' or 'desc'", style="bold")
        raise typer.Exit(1)

    # Parse filter - supports two modes:
    #   1) comma-separated keys (filter by key existence)
    #   2) JSON object (filter by key-value pairs)
    filter_mode: Optional[str] = None  # 'keys' or 'key_values'
    filter_keys: Optional[List[str]] = None
    filter_dict: Optional[Dict[str, Any]] = None

    if filter_json:
        # Try to parse as JSON first (mode 2)
        try:
            parsed = json.loads(filter_json)
            if isinstance(parsed, dict):
                filter_mode = 'key_values'
                filter_dict = parsed
            else:
                console.print("[red]✗[/red] Filter must be a JSON object or comma-separated keys", style="bold")
                raise typer.Exit(1)
        except json.JSONDecodeError:
            # Not JSON, treat as comma-separated keys (mode 1)
            filter_mode = 'keys'
            filter_keys = [k.strip() for k in filter_json.split(',') if k.strip()]
            if not filter_keys:
                console.print("[red]✗[/red] Empty filter value", style="bold")
                raise typer.Exit(1)
    
    # Get documents from server
    try:
        response = requests.get(
            f"http://127.0.0.1:{port}/documents/list",
            timeout=10
        )
        
        if response.status_code != 200:
            console.print("[red]✗[/red] Failed to list documents", style="bold")
            raise typer.Exit(1)
        
        data = response.json()
        documents = data['documents']
    
    except requests.RequestException as e:
        console.print(f"[red]✗[/red] Failed to communicate with server: {e}", style="bold")
        raise typer.Exit(1)
    
    if not documents:
        console.print("No documents in database")
        return
    
    # Apply filter if provided
    if filter_mode == 'keys' and filter_keys is not None:
        # Mode 1: Filter by key existence
        filtered_docs = []
        for doc in documents:
            doc_metadata = doc.get('metadata', {})
            if not doc_metadata:
                continue
            # Check if all required keys exist in metadata
            if all(k in doc_metadata for k in filter_keys):
                filtered_docs.append(doc)
        documents = filtered_docs

    elif filter_mode == 'key_values' and filter_dict is not None:
        # Mode 2: Filter by key-value pairs
        filtered_docs = []
        for doc in documents:
            doc_metadata = doc.get('metadata', {})
            if not doc_metadata:
                continue
            # Check if all filter key-value pairs match
            if all(doc_metadata.get(k) == v for k, v in filter_dict.items()):
                filtered_docs.append(doc)
        documents = filtered_docs

    # Handle no results after filtering
    if not documents and filter_mode is not None:
        if filter_mode == 'keys':
            if filter_keys:
                console.print(f"[yellow]No documents match the filter keys: {', '.join(filter_keys)}[/yellow]")
                console.print("\n[dim]Hint: Documents must have all these metadata keys.[/dim]")
                console.print(
                    f"  [dim]Example: rag-anywhere add <file> --metadata '[/dim]"
                    + '{"' + filter_keys[0] + '":"value"}'
                    + "[dim]'[/dim]"
                )
            else:
                # Should not happen because we exit on empty filter_keys above,
                # but kept for safety and clearer typing.
                console.print("[yellow]No documents match the filter keys[/yellow]")
        elif filter_mode == 'key_values':
            if filter_dict is not None:
                console.print(f"[yellow]No documents match the filter: {json.dumps(filter_dict)}[/yellow]")
            else:
                console.print("[yellow]No documents match the filter[/yellow]")
            console.print("\n[dim]Hint: Make sure you added documents with matching metadata using:[/dim]")
            console.print(f"  [dim]rag-anywhere add <file> --metadata '[/dim]" + '{"key":"value"}' + "[dim]'[/dim]")
        return

    if not documents:
        console.print("No documents in database")
        return

    # Sort documents
    if sort_by == 'created':
        documents.sort(key=lambda x: x['created_at'], reverse=(order == 'desc'))
    elif sort_by == 'filename':
        documents.sort(key=lambda x: x['filename'].lower(), reverse=(order == 'desc'))
    elif sort_by == 'chunks':
        documents.sort(key=lambda x: x.get('num_chunks', 0), reverse=(order == 'desc'))

    # Calculate pagination
    total_docs = len(documents)
    total_pages = (total_docs + per_page - 1) // per_page  # Ceiling division

    if page > total_pages and total_pages > 0:
        console.print(f"[yellow]⚠[/yellow]  Page {page} exceeds total pages ({total_pages}). Showing last page.", style="bold")
        page = total_pages

    # Apply pagination
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_docs)
    paginated_docs = documents[start_idx:end_idx]

    # Display documents
    if verbose:
        for doc in paginated_docs:
            console.print(f"\n[bold cyan]{doc['filename']}[/bold cyan]")
            console.print(f"  ID: {doc['id']}")
            console.print(f"  Created: {doc['created_at']}")
            if doc.get('metadata'):
                console.print(f"  Metadata: {json.dumps(doc['metadata'])}")
            console.print(f"  Chunks: {doc.get('num_chunks', 'N/A')}")
    else:
        table = Table(title=f"Documents in '{db_name}' (Page {page}/{total_pages})")
        table.add_column("Filename", style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Created", style="magenta")
        table.add_column("Chunks", style="green")

        for doc in paginated_docs:
            table.add_row(
                doc['filename'],
                doc['id'][:8] + "...",
                doc['created_at'][:10],  # Just the date
                str(doc.get('num_chunks', 'N/A'))
            )

        console.print(table)

        # Pagination info
        console.print(f"\nShowing {start_idx + 1}-{end_idx} of {total_docs} document(s)")
        if page < total_pages:
            console.print(f"[dim]Use --page {page + 1} to see more[/dim]")
