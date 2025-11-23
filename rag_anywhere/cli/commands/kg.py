"""Knowledge Graph CLI commands."""

import typer
from rich.console import Console
from rich.table import Table
from rich.json import JSON
from typing import Optional, List
import json

from ..context import RAGContext

app = typer.Typer(help="Knowledge graph operations")
console = Console()


@app.command("list-entities")
def list_entities(
    category: Optional[str] = typer.Option(None, "--type", "-t", help="Filter by entity category"),
    min_freq: Optional[int] = typer.Option(None, "--min-freq", "-m", help="Minimum frequency"),
    limit: Optional[int] = typer.Option(50, "--limit", "-l", help="Maximum number of results"),
):
    """List entities from the knowledge graph."""
    ctx = RAGContext()

    try:
        ctx.ensure_active_database()
        db_name = ctx.get_active_database_name()
        # Load only the EntityStore without heavy models
        entity_store = ctx.load_entity_store_only(db_name)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    # Query entities
    entities = entity_store.query_entities(
        category=category,
        min_frequency=min_freq,
        limit=limit
    )

    if not entities:
        console.print("[yellow]No entities found[/yellow]")
        return

    # Create table
    table = Table(title=f"Knowledge Graph Entities ({len(entities)} results)")
    table.add_column("ID", style="cyan")
    table.add_column("Display Name", style="green")
    table.add_column("Category", style="blue")
    table.add_column("Frequency", justify="right", style="magenta")

    for entity in entities:
        table.add_row(
            str(entity['id']),
            entity['display_name'],
            entity['category'],
            str(entity['frequency'])
        )

    console.print(table)


@app.command("show-entity")
def show_entity(
    name: str = typer.Argument(..., help="Entity name to look up"),
    category: Optional[str] = typer.Option(None, "--type", "-t", help="Entity category"),
    show_chunks: bool = typer.Option(False, "--chunks", "-c", help="Show related chunks"),
):
    """Show details about a specific entity."""
    ctx = RAGContext()

    try:
        ctx.ensure_active_database()
        db_name = ctx.get_active_database_name()
        # Load only the EntityStore without heavy models
        entity_store = ctx.load_entity_store_only(db_name)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    # Get entity
    entity = entity_store.get_entity_by_name(name, category)

    if not entity:
        console.print(f"[red]Entity '{name}' not found[/red]")
        if category:
            console.print(f"[dim](searched in category: {category})[/dim]")
        return

    # Display entity details
    console.print(f"\n[bold]Entity Details:[/bold]")
    console.print(f"  ID: {entity['id']}")
    console.print(f"  Name: {entity['display_name']}")
    console.print(f"  Category: {entity['category']}")
    console.print(f"  Frequency: {entity['frequency']}")

    # Get related chunks
    chunk_ids = entity_store.get_entity_chunks(name, category)
    console.print(f"\n[bold]Found in {len(chunk_ids)} chunks[/bold]")

    if show_chunks and chunk_ids:
        console.print()
        for i, chunk_id in enumerate(chunk_ids[:10], 1):  # Show first 10
            console.print(f"  {i}. {chunk_id}")
        if len(chunk_ids) > 10:
            console.print(f"  [dim]... and {len(chunk_ids) - 10} more[/dim]")

    # Get related entities (co-occurrence)
    related = entity_store.get_related_entities(entity['id'], limit=10)
    if related:
        console.print(f"\n[bold]Related Entities (top 10 by co-occurrence):[/bold]")
        for rel_entity, count in related:
            console.print(f"  • {rel_entity['display_name']} ({rel_entity['category']}) - "
                         f"{count} shared chunks")


@app.command("stats")
def show_stats():
    """Show knowledge graph statistics."""
    ctx = RAGContext()

    try:
        ctx.ensure_active_database()
        db_name = ctx.get_active_database_name()
        # Load only the EntityStore without heavy models
        entity_store = ctx.load_entity_store_only(db_name)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    # Get stats
    stats = entity_store.get_stats()

    console.print(f"\n[bold]Knowledge Graph Statistics for '{db_name}':[/bold]\n")
    console.print(f"  Total Entities: {stats['total_entities']}")
    console.print(f"  Total Edges: {stats['total_edges']}")

    # By category
    if stats['by_category']:
        console.print(f"\n[bold]Entities by Category:[/bold]")
        table = Table(show_header=True)
        table.add_column("Category", style="blue")
        table.add_column("Count", justify="right", style="cyan")

        for category, count in sorted(stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            table.add_row(category, str(count))

        console.print(table)

    # Top entities
    if stats['top_entities']:
        console.print(f"\n[bold]Top Entities (by frequency):[/bold]")
        table = Table(show_header=True)
        table.add_column("Entity", style="green")
        table.add_column("Category", style="blue")
        table.add_column("Frequency", justify="right", style="magenta")

        for entity in stats['top_entities']:
            table.add_row(
                entity['display_name'],
                entity['category'],
                str(entity['frequency'])
            )

        console.print(table)


@app.command("reprocess")
def reprocess_document(
    doc_id: Optional[str] = typer.Argument(None, help="Document ID to reprocess (omit for --all)"),
    all_docs: bool = typer.Option(False, "--all", "-a", help="Reprocess all documents"),
    labels: Optional[str] = typer.Option(None, "--labels", "-l", help="Comma-separated additional labels"),
):
    """Reprocess document(s) to extract entities."""
    ctx = RAGContext()

    try:
        ctx.ensure_active_database()
        db_name = ctx.get_active_database_name()
        ctx.load_database(db_name, verbose=False)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    if ctx.entity_store is None:
        console.print("[yellow]Knowledge graph is disabled for this database[/yellow]")
        return

    if not all_docs and not doc_id:
        console.print("[red]Error: Provide document ID or use --all flag[/red]")
        raise typer.Exit(1)

    # Parse labels
    user_labels = []
    if labels:
        user_labels = [label.strip() for label in labels.split(',')]

    if all_docs:
        console.print("[yellow]Reprocessing all documents...[/yellow]")
        docs = ctx.safe_document_store.list_documents()
        doc_ids = [doc['id'] for doc in docs]
    else:
        doc_ids = [doc_id]

    # Get GLiNER config
    gliner_config = ctx.db_config.get('gliner', {})
    default_labels = gliner_config.get('default_labels', [])

    total_entities = 0
    for did in doc_ids:
        console.print(f"Processing document: {did}")

        # Get chunks for document
        chunks = ctx.safe_document_store.get_chunks_by_document(did)

        if not chunks:
            console.print(f"  [yellow]No chunks found for document {did}[/yellow]")
            continue

        # Convert to TextChunk-like objects for processing
        from ...core.splitters.base import TextChunk

        text_chunks = []
        for chunk in chunks:
            tc = TextChunk(
                content=chunk['content'],
                start_char=chunk['start_char'],
                end_char=chunk['end_char'],
                metadata={
                    'document_id': did,
                    'chunk_index': chunk['chunk_index']
                }
            )
            text_chunks.append(tc)

        # Process with GLiNER
        chunk_entities_map = ctx.safe_gliner_processor.process_chunks(
            text_chunks,
            default_labels,
            user_labels
        )

        # Delete existing entities and add new ones
        for chunk_id, chunk_entities in chunk_entities_map.items():
            ctx.safe_entity_store.delete_chunk_entities(chunk_id)
            num_entities = ctx.safe_entity_store.add_entities(
                chunk_id,
                chunk_entities.entities,
                source='gliner'
            )
            total_entities += num_entities

        console.print(f"  ✓ Extracted {len(chunk_entities_map)} chunk entity sets")

    console.print(f"\n[green]✓ Reprocessing complete. Total entities: {total_entities}[/green]")


@app.command("export")
def export_graph(
    output: str = typer.Argument(..., help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json)"),
):
    """Export knowledge graph to file."""
    ctx = RAGContext()
    ctx.ensure_active_database()

    db_name = ctx.get_active_database_name()

    try:
        # Load only the EntityStore without heavy models
        entity_store = ctx.load_entity_store_only(db_name)
    except ValueError as e:
        console.print(f"[yellow]{e}[/yellow]")
        return

    if format != "json":
        console.print(f"[red]Error: Unsupported format '{format}'. Only 'json' is currently supported.[/red]")
        raise typer.Exit(1)

    # Export all entities and stats
    entities = entity_store.query_entities(limit=None)
    stats = entity_store.get_stats()

    export_data = {
        "database": db_name,
        "stats": stats,
        "entities": entities
    }

    with open(output, 'w') as f:
        json.dump(export_data, f, indent=2)

    console.print(f"[green]✓ Exported {len(entities)} entities to {output}[/green]")


if __name__ == "__main__":
    app()
