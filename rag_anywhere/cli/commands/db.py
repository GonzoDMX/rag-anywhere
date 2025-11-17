# rag_anywhere/cli/commands/db.py
import typer
from rich.console import Console
from rich.table import Table
from pathlib import Path
from typing import Optional
import os

from ..context import RAGContext
from ...core.embeddings.providers import EmbeddingGemmaProvider, OpenAIEmbeddingProvider

app = typer.Typer()
console = Console()


@app.command()
def create(
    name: str = typer.Argument(..., help="Database name"),
    provider: str = typer.Option(
        "embeddinggemma",
        "--provider",
        "-p",
        help="Embedding provider (embeddinggemma, openai)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name (default depends on provider)"
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        "-d",
        help="Device for local models (auto, cpu, cuda, mps)"
    )
):
    """Create a new database"""
    ctx = RAGContext()
    
    # Check if database already exists
    if ctx.config.database_exists(name):
        console.print(f"[red]✗[/red] Database '{name}' already exists", style="bold")
        raise typer.Exit(1)
    
    # Validate database name
    if not name.replace('-', '').replace('_', '').isalnum():
        console.print(
            "[red]✗[/red] Database name must contain only letters, numbers, hyphens, and underscores",
            style="bold"
        )
        raise typer.Exit(1)
    
    # Set default models
    if model is None:
        if provider == "embeddinggemma":
            model = "google/embeddinggemma-300m"
        elif provider == "openai":
            model = "text-embedding-3-small"
        else:
            console.print(f"[red]✗[/red] Unknown provider: {provider}", style="bold")
            raise typer.Exit(1)
    
    # Get model specifications
    console.print(f"Creating database '{name}' with {provider}:{model}...")
    
    try:
        if provider == "embeddinggemma":
            # Create a temporary provider to get specs
            temp_provider = EmbeddingGemmaProvider(model_name=model, device=device)
            dimension = temp_provider.dimension
            max_tokens = temp_provider.max_tokens
            
            # Clean up
            del temp_provider
            
        elif provider == "openai":
            # Check for API key
            if not os.environ.get('OPENAI_API_KEY'):
                console.print(
                    "[red]✗[/red] OpenAI provider requires OPENAI_API_KEY environment variable",
                    style="bold"
                )
                raise typer.Exit(1)
            
            dimension = 768  # We configure OpenAI to return 768 dimensions
            max_tokens = 8191
        
        else:
            console.print(f"[red]✗[/red] Unsupported provider: {provider}", style="bold")
            raise typer.Exit(1)
        
        # Create database config
        db_config = ctx.config.create_database_config(
            db_name=name,
            embedding_provider=provider,
            embedding_model=model,
            embedding_dimension=dimension,
            embedding_max_tokens=max_tokens,
            additional_config={
                'embedding': {'device': device} if provider == 'embeddinggemma' else {}
            }
        )
        
        # Load the database (creates SQLite DB and loads model)
        ctx.load_database(name, verbose=False)
        
        console.print(f"[green]✓[/green] Created database '{name}'", style="bold")
        console.print(f"  Provider: {provider}")
        console.print(f"  Model: {model}")
        console.print(f"  Dimensions: {dimension}")
        console.print(f"  Max tokens: {max_tokens}")
        console.print(f"  Database is now [green]ACTIVE[/green]")
        
    except Exception as e:
        # Clean up on failure
        if ctx.config.database_exists(name):
            ctx.config.delete_database(name)
        console.print(f"[red]✗[/red] Error creating database: {e}", style="bold")
        raise typer.Exit(1)


@app.command()
def list():
    """List all databases"""
    ctx = RAGContext()
    db_names = ctx.config.list_databases()
    
    if not db_names:
        console.print("No databases found. Create one with 'rag-anywhere db create <name>'")
        return
    
    active_db = ctx.config.get_active_database()
    
    table = Table(title="Databases")
    table.add_column("Name", style="cyan")
    table.add_column("Provider", style="magenta")
    table.add_column("Model", style="blue")
    table.add_column("Status", style="green")
    
    for db_name in sorted(db_names):
        try:
            config = ctx.config.load_database_config(db_name)
            provider = config['embedding']['provider']
            model = config['embedding']['model']
            status = "ACTIVE" if db_name == active_db else ""
            
            table.add_row(db_name, provider, model, status)
        except Exception:
            table.add_row(db_name, "ERROR", "ERROR", "")
    
    console.print(table)


@app.command()
def use(name: str = typer.Argument(..., help="Database name")):
    """Switch to a different database"""
    ctx = RAGContext()
    
    if not ctx.config.database_exists(name):
        console.print(f"[red]✗[/red] Database '{name}' does not exist", style="bold")
        console.print("Available databases:")
        for db_name in ctx.config.list_databases():
            console.print(f"  - {db_name}")
        raise typer.Exit(1)
    
    try:
        ctx.load_database(name, verbose=True)
    except Exception as e:
        console.print(f"[red]✗[/red] Error loading database: {e}", style="bold")
        raise typer.Exit(1)


@app.command()
def info(name: Optional[str] = typer.Argument(None, help="Database name (defaults to active)")):
    """Show detailed information about a database"""
    ctx = RAGContext()
    
    # Determine which database to show info for
    if name is None:
        name = ctx.config.get_active_database()
        if name is None:
            console.print("[red]✗[/red] No active database", style="bold")
            raise typer.Exit(1)
    
    if not ctx.config.database_exists(name):
        console.print(f"[red]✗[/red] Database '{name}' does not exist", style="bold")
        raise typer.Exit(1)
    
    # Load config
    config = ctx.config.load_database_config(name)
    db_path = ctx.config.get_database_db_path(name)
    
    # Get database statistics
    from ...core import DocumentStore, VectorStore
    doc_store = DocumentStore(str(db_path))
    vec_store = VectorStore(str(db_path), dimension=config['embedding']['dimension'])
    
    documents = doc_store.list_documents()
    num_chunks = len(doc_store.get_all_chunk_ids())
    
    # Calculate storage size
    storage_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0
    
    # Display info
    is_active = (name == ctx.config.get_active_database())
    status = "[green]ACTIVE[/green]" if is_active else ""
    
    console.print(f"\n[bold]Database: {name}[/bold] {status}")
    console.print(f"[dim]{'─' * 60}[/dim]")
    
    console.print(f"\n[bold]Embedding Model:[/bold]")
    console.print(f"  Provider: {config['embedding']['provider']}")
    console.print(f"  Model: {config['embedding']['model']}")
    console.print(f"  Dimensions: {config['embedding']['dimension']}")
    console.print(f"  Max tokens: {config['embedding']['max_tokens']}")
    
    console.print(f"\n[bold]Content:[/bold]")
    console.print(f"  Documents: {len(documents)}")
    console.print(f"  Chunks: {num_chunks}")
    console.print(f"  Vectors: {vec_store.count()}")
    
    console.print(f"\n[bold]Storage:[/bold]")
    console.print(f"  Database size: {storage_mb:.2f} MB")
    console.print(f"  Location: {db_path}")
    
    console.print(f"\n[bold]Configuration:[/bold]")
    console.print(f"  Created: {config['database']['created_at']}")
    console.print(f"  Version: {config['database']['version']}")
    
    console.print()


@app.command()
def delete(
    name: str = typer.Argument(..., help="Database name"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation")
):
    """Delete a database and all its contents"""
    ctx = RAGContext()
    
    if not ctx.config.database_exists(name):
        console.print(f"[red]✗[/red] Database '{name}' does not exist", style="bold")
        raise typer.Exit(1)
    
    # Confirmation
    if not force:
        console.print(f"[yellow]⚠[/yellow]  This will permanently delete database '{name}' and all its contents.")
        confirm = typer.confirm("Continue?")
        if not confirm:
            console.print("Cancelled")
            raise typer.Exit(0)
    
    try:
        ctx.config.delete_database(name)
        console.print(f"[green]✓[/green] Deleted database '{name}'", style="bold")
    except Exception as e:
        console.print(f"[red]✗[/red] Error deleting database: {e}", style="bold")
        raise typer.Exit(1)


@app.command()
def rename(
    old_name: str = typer.Argument(..., help="Current database name"),
    new_name: str = typer.Argument(..., help="New database name")
):
    """Rename a database"""
    ctx = RAGContext()
    
    if not ctx.config.database_exists(old_name):
        console.print(f"[red]✗[/red] Database '{old_name}' does not exist", style="bold")
        raise typer.Exit(1)
    
    if ctx.config.database_exists(new_name):
        console.print(f"[red]✗[/red] Database '{new_name}' already exists", style="bold")
        raise typer.Exit(1)
    
    # Validate new name
    if not new_name.replace('-', '').replace('_', '').isalnum():
        console.print(
            "[red]✗[/red] Database name must contain only letters, numbers, hyphens, and underscores",
            style="bold"
        )
        raise typer.Exit(1)
    
    try:
        import shutil
        old_dir = ctx.config.get_database_dir(old_name)
        new_dir = ctx.config.get_database_dir(new_name)
        
        shutil.move(str(old_dir), str(new_dir))
        
        # Update config
        config = ctx.config.load_database_config(new_name)
        config['database']['name'] = new_name
        ctx.config.save_database_config(new_name, config)
        
        # Update active database if needed
        if ctx.config.get_active_database() == old_name:
            ctx.config.set_active_database(new_name)
        
        console.print(f"[green]✓[/green] Renamed database '{old_name}' to '{new_name}'", style="bold")
    except Exception as e:
        console.print(f"[red]✗[/red] Error renaming database: {e}", style="bold")
        raise typer.Exit(1)
