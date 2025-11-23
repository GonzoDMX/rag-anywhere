"""Knowledge Graph API routes."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional, List

from ..models import (
    EntityItem,
    EntityListResponse,
    EntityDetailsResponse,
    KGStatsResponse,
    ReprocessRequest,
    ReprocessResponse,
)
from ..dependencies import get_rag_context_with_database
from ...cli.context import RAGContext

router = APIRouter(prefix="/kg", tags=["knowledge-graph"])


@router.get("/entities", response_model=EntityListResponse)
async def list_entities(
    category: Optional[str] = Query(None, description="Filter by entity category"),
    min_frequency: Optional[int] = Query(None, description="Minimum frequency"),
    limit: int = Query(50, ge=1, le=1000, description="Maximum results"),
    rag_context: RAGContext = Depends(get_rag_context_with_database),
):
    """
    List entities from the knowledge graph.

    - **category**: Filter by entity type (person, organization, location, etc.)
    - **min_frequency**: Minimum number of mentions
    - **limit**: Maximum number of results to return
    """
    if rag_context.entity_store is None:
        raise HTTPException(
            status_code=400, detail="Knowledge graph is disabled for this database"
        )

    try:
        entities = rag_context.safe_entity_store.query_entities(
            category=category, min_frequency=min_frequency, limit=limit
        )

        return EntityListResponse(
            entities=[EntityItem(**entity) for entity in entities],
            total=len(entities),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_name}", response_model=EntityDetailsResponse)
async def get_entity_details(
    entity_name: str,
    category: Optional[str] = Query(None, description="Entity category"),
    rag_context: RAGContext = Depends(get_rag_context_with_database),
):
    """
    Get detailed information about a specific entity.

    - **entity_name**: Name of the entity (case-insensitive)
    - **category**: Optional category filter
    """
    if rag_context.entity_store is None:
        raise HTTPException(
            status_code=400, detail="Knowledge graph is disabled for this database"
        )

    try:
        entity = rag_context.safe_entity_store.get_entity_by_name(entity_name, category)

        if not entity:
            raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")

        # Get related chunks
        chunk_ids = rag_context.safe_entity_store.get_entity_chunks(entity_name, category)

        # Get related entities (co-occurrence)
        related = rag_context.safe_entity_store.get_related_entities(entity['id'], limit=20)
        related_entities = [
            {
                "entity": {
                    "id": rel_ent["id"],
                    "name": rel_ent["name"],
                    "display_name": rel_ent["display_name"],
                    "category": rel_ent["category"],
                    "frequency": rel_ent["frequency"],
                },
                "co_occurrence_count": count,
            }
            for rel_ent, count in related
        ]

        return EntityDetailsResponse(
            entity=EntityItem(**entity),
            chunk_ids=chunk_ids,
            related_entities=related_entities,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chunks", response_model=List[str])
async def get_chunks_by_entity(
    entity: str = Query(..., description="Entity name"),
    category: Optional[str] = Query(None, description="Entity category"),
    rag_context: RAGContext = Depends(get_rag_context_with_database),
):
    """
    Get all chunk IDs that mention a specific entity.

    - **entity**: Entity name to search for
    - **category**: Optional category filter
    """
    if rag_context.entity_store is None:
        raise HTTPException(
            status_code=400, detail="Knowledge graph is disabled for this database"
        )

    try:
        chunk_ids = rag_context.safe_entity_store.get_entity_chunks(entity, category)
        return chunk_ids
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=KGStatsResponse)
async def get_stats(
    rag_context: RAGContext = Depends(get_rag_context_with_database),
):
    """
    Get knowledge graph statistics.

    Returns entity counts, category breakdown, and top entities.
    """
    if rag_context.entity_store is None:
        raise HTTPException(
            status_code=400, detail="Knowledge graph is disabled for this database"
        )

    try:
        stats = rag_context.safe_entity_store.get_stats()
        return KGStatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reprocess", response_model=ReprocessResponse)
async def reprocess_entities(
    request: ReprocessRequest,
    rag_context: RAGContext = Depends(get_rag_context_with_database),
):
    """
    Reprocess document(s) to re-extract entities.

    - **document_id**: Specific document to reprocess (omit to reprocess all)
    - **labels**: Additional entity labels beyond defaults
    """
    if rag_context.entity_store is None:
        raise HTTPException(
            status_code=400, detail="Knowledge graph is disabled for this database"
        )

    if rag_context.gliner_processor is None:
        raise HTTPException(
            status_code=400, detail="GLiNER processor not available"
        )

    try:
        # Get documents to process
        if request.document_id:
            doc_ids = [request.document_id]
        else:
            docs = rag_context.safe_document_store.list_documents()
            doc_ids = [doc['id'] for doc in docs]

        # Get GLiNER config
        gliner_config = rag_context.db_config.get('gliner', {})
        default_labels = gliner_config.get('default_labels', [])
        user_labels = request.labels or []

        total_entities = 0
        for doc_id in doc_ids:
            # Get chunks for document
            chunks = rag_context.safe_document_store.get_chunks_by_document(doc_id)

            if not chunks:
                continue

            # Convert to TextChunk-like objects
            from ...core.splitters.base import TextChunk

            text_chunks = []
            for chunk in chunks:
                tc = TextChunk(
                    content=chunk['content'],
                    start_char=chunk['start_char'],
                    end_char=chunk['end_char'],
                    metadata={'document_id': doc_id, 'chunk_index': chunk['chunk_index']},
                )
                text_chunks.append(tc)

            # Process with GLiNER
            chunk_entities_map = rag_context.safe_gliner_processor.process_chunks(
                text_chunks, default_labels, user_labels
            )

            # Delete existing and add new
            for chunk_id, chunk_entities in chunk_entities_map.items():
                rag_context.safe_entity_store.delete_chunk_entities(chunk_id)
                num_entities = rag_context.safe_entity_store.add_entities(
                    chunk_id, chunk_entities.entities, source='gliner'
                )
                total_entities += num_entities

        return ReprocessResponse(
            status="success",
            documents_processed=len(doc_ids),
            total_entities=total_entities,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
