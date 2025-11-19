# rag_anywhere/cli/commands/db.py

import os
import sys
import platform
import typer
from rich.console import Console
from rich.table import Table
from typing import Optional
from pathlib import Path

from ..context import RAGContext
from ...server.manager import ServerManager
from ...server.state import ServerStatus
from ...utils.logging import get_logger

app = typer.Typer()
console = Console()
logger = get_logger('cli.db')


def _validate_local_model_path(model_path: str) -> bool:
    """
    Validate that a local model path exists and contains required files.

    Returns:
        True if valid, False otherwise
    """
    path = Path(model_path).expanduser().resolve()

    if not path.exists():
        logger.error(f"Model path does not exist: {path}")
        return False

    if not path.is_dir():
        logger.error(f"Model path is not a directory: {path}")
        return False

    # Check for essential model files
    required_files = ['config.json']
    has_model_file = False

    for file in required_files:
        if not (path / file).exists():
            logger.error(f"Missing required file '{file}' in model directory: {path}")
            return False

    # Check for at least one model weight file
    model_extensions = ['.bin', '.safetensors', '.pt', '.pth']
    for ext in model_extensions:
        if any(path.glob(f'*{ext}')):
            has_model_file = True
            break

    if not has_model_file:
        logger.error(f"No model weight files found in: {path}")
        return False

    logger.info(f"Local model path validated: {path}")
    return True


def _is_local_path(model: str) -> bool:
    """Check if model string is a local path (relative or absolute)."""
    return model.startswith(('.', '/', '~')) or Path(model).exists()


@app.command()
def create(
    name: str = typer.Argument(..., help="Database name"),
    provider: str = typer.Option(
        "embeddinggemma",
        "--provider",
        "-p",
        help="Embedding provider: embeddinggemma (local) or openai (API)"
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name or path. For embeddinggemma: HuggingFace model name or local path. For openai: model name (text-embedding-3-small, text-embedding-3-large)"
    )
):
    """Create a new database"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
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

    # Log system information for debugging
    logger.info(f"Creating database '{name}'")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Machine: {platform.machine()}")
    logger.info(f"Provider: {provider}, Model: {model}")

    # Initialize model_to_store (will be updated if local model is cached)
    model_to_store = model

    try:
        if provider == "embeddinggemma":
            logger.info("Validating EmbeddingGemma provider configuration...")

            # Check dependencies are available (without importing heavy modules)
            try:
                import importlib.util
                if importlib.util.find_spec("torch") is None:
                    raise ImportError("torch not found")
                if importlib.util.find_spec("sentence_transformers") is None:
                    raise ImportError("sentence_transformers not found")
            except ImportError as e:
                logger.error(f"Missing required dependencies: {e}")
                console.print(
                    f"[red]✗[/red] Missing dependencies. Install with: pip install torch sentence-transformers",
                    style="bold"
                )
                raise typer.Exit(1)

            # Validate and cache model if needed
            try:
                # Check if it's a local path
                if _is_local_path(model):
                    logger.info(f"Validating local model path: {model}")
                    if not _validate_local_model_path(model):
                        console.print(
                            f"[red]✗[/red] Invalid local model path: {model}",
                            style="bold"
                        )
                        console.print("Ensure the directory contains config.json and model weight files")
                        raise typer.Exit(1)

                    console.print(f"[green]✓[/green] Local model validated: {model}")

                    # Cache the local model
                    console.print("Copying model to cache...")
                    logger.info(f"Caching local model from {model}")
                    cached_path = ctx.config.cache_local_model(model)
                    model_to_store = cached_path
                    logger.info(f"Model cached to: {cached_path}")
                    console.print(f"[green]✓[/green] Model cached to: {cached_path}")
                else:
                    # For HuggingFace models, just check if it might be cached
                    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                    model_cache_name = f"models--{model.replace('/', '--')}"
                    model_cached = (cache_dir / model_cache_name).exists()

                    if model_cached:
                        logger.info(f"Model '{model}' found in cache")
                        console.print(f"[green]✓[/green] Model '{model}' found in cache")
                    else:
                        logger.info(f"Model '{model}' will be downloaded on first use (~1.2GB)")
                        console.print(f"[yellow]ℹ[/yellow] Model '{model}' will be downloaded when server starts")

                dimension = 768  # EmbeddingGemma dimension
                max_tokens = 2048

            except typer.Exit:
                raise
            except Exception as e:
                logger.error(f"Failed to validate model: {type(e).__name__}: {e}", exc_info=True)
                console.print(
                    f"[red]✗[/red] Failed to validate embedding model",
                    style="bold"
                )
                console.print(f"Error: {e}")
                raise typer.Exit(1)
            
        elif provider == "openai":
            logger.info("Initializing OpenAI provider...")

            # Check for API key
            if not os.environ.get('OPENAI_API_KEY'):
                logger.error("OPENAI_API_KEY environment variable not set")
                console.print(
                    "[red]✗[/red] OpenAI provider requires OPENAI_API_KEY environment variable",
                    style="bold"
                )
                raise typer.Exit(1)

            dimension = 768
            max_tokens = 8191
            logger.info("OpenAI provider configured")

        else:
            logger.error(f"Unsupported provider: {provider}")
            console.print(f"[red]✗[/red] Unsupported provider: {provider}", style="bold")
            raise typer.Exit(1)

        # Create database config
        logger.info("Creating database configuration...")
        db_config = ctx.config.create_database_config(
            db_name=name,
            embedding_provider=provider,
            embedding_model=model_to_store,
            embedding_dimension=dimension,
            embedding_max_tokens=max_tokens
        )

        if not db_config:
            logger.error("Failed to create database configuration")
            console.print(f"[red]✗[/red] Failed to create database configuration", style="bold")
            raise typer.Exit(1)

        logger.info("Database configuration created successfully")

        # Set as active database
        logger.info(f"Setting '{name}' as active database")
        ctx.config.set_active_database(name)

        console.print(f"[green]✓[/green] Created database '{name}'", style="bold")
        console.print(f"  Provider: {provider}")
        console.print(f"  Model: {model}")
        console.print(f"  Dimensions: {dimension}")
        console.print(f"  Max tokens: {max_tokens}")

        # Start server with new database
        console.print(f"\nStarting server for database '{name}'...")
        try:
            logger.info("Starting server...")
            manager.start_server()
            logger.info("Server started successfully")
            console.print(f"[green]✓[/green] Server started", style="bold")
        except Exception as e:
            logger.warning(f"Server start failed: {e}", exc_info=True)
            console.print(f"[yellow]⚠[/yellow]  Server start failed: {e}")
            console.print("You can start it manually with: rag-anywhere server start")

    except typer.Exit:
        # Re-raise typer.Exit to allow clean exit
        raise
    except Exception as e:
        # Clean up on failure
        logger.error(f"Database creation failed: {type(e).__name__}: {e}", exc_info=True)
        if ctx.config.database_exists(name):
            logger.info(f"Cleaning up failed database '{name}'")
            ctx.config.delete_database(name)
        console.print(f"[red]✗[/red] Error creating database: {e}", style="bold")
        console.print("\n[yellow]Tip:[/yellow] Run with --debug flag for detailed logs:")
        console.print(f"  rag-anywhere --debug db create {name}")
        raise typer.Exit(1)


@app.command()
def use(name: str = typer.Argument(..., help="Database name")):
    """Switch to a different database"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
    if not ctx.config.database_exists(name):
        console.print(f"[red]✗[/red] Database '{name}' does not exist", style="bold")
        console.print("Available databases:")
        for db_name in ctx.config.list_databases():
            console.print(f"  - {db_name}")
        raise typer.Exit(1)
    
    # Get current active database
    current_db = ctx.config.get_active_database()
    
    if current_db == name:
        console.print(f"Database '{name}' is already active")
        return
    
    # Warn if switching from another database
    if current_db:
        console.print(f"[yellow]⚠[/yellow]  Switching from '{current_db}' to '{name}'")
    
    try:
        # Set as active
        ctx.config.set_active_database(name)
        
        # Check server status
        status = manager.get_status()
        
        if status['status'] in [ServerStatus.RUNNING.value, ServerStatus.SLEEPING.value]:
            # Server is running, need to switch database
            console.print("Reloading server with new database...")
            
            success = manager.switch_database(name)
            
            if success:
                console.print(f"[green]✓[/green] Switched to database '{name}'", style="bold")
            else:
                # Fallback: restart server
                console.print("[yellow]Server reload failed, restarting...[/yellow]")
                manager.restart_server()
                console.print(f"[green]✓[/green] Switched to database '{name}'", style="bold")
        else:
            # Server not running, just set active and start
            console.print(f"Starting server for database '{name}'...")
            manager.start_server()
            console.print(f"[green]✓[/green] Switched to database '{name}'", style="bold")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Error switching database: {e}", style="bold")
        raise typer.Exit(1)


@app.command()
def deactivate():
    """Deactivate the current database and stop the server"""
    ctx = RAGContext()
    manager = ServerManager(ctx.config)
    
    active_db = ctx.config.get_active_database()
    
    if not active_db:
        console.print("No active database")
        return
    
    console.print(f"Deactivating database '{active_db}'...")
    
    # Stop server
    status = manager.get_status()
    if status['status'] != ServerStatus.STOPPED.value:
        console.print("Stopping server...")
        manager.stop_server()
    
    # Clear active database
    global_config = ctx.config.load_global_config()
    global_config.pop('active_database', None)
    ctx.config.save_global_config(global_config)
    
    console.print(f"[green]✓[/green] Deactivated database '{active_db}'", style="bold")


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
    manager = ServerManager(ctx.config)
    
    if not ctx.config.database_exists(name):
        console.print(f"[red]✗[/red] Database '{name}' does not exist", style="bold")
        raise typer.Exit(1)
    
    # Check if database is active
    active_db = ctx.config.get_active_database()
    if active_db == name:
        console.print(
            f"[red]✗[/red] Cannot delete active database '{name}'",
            style="bold"
        )
        console.print("Deactivate it first with: rag-anywhere db deactivate")
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
