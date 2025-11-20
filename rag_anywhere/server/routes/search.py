# rag_anywhere/server/routes/search.py

from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    SearchRequest, SearchResponse, SearchResultItem,
    KeywordSearchRequest, KeywordSearchResponse, KeywordSearchResultItem,
    AdvancedKeywordSearchRequest,
    DocumentInfo, ChunkPosition
)
from ..dependencies import get_rag_context_with_database
from ...cli.context import RAGContext

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    rag_context: RAGContext = Depends(get_rag_context_with_database)
):
    """
    Search for documents using semantic similarity.
    
    - **query**: The search query text
    - **top_k**: Number of results to return (1-100)
    - **min_score**: Minimum similarity score threshold (0.0-1.0)
    """
    try:
        results = rag_context.safe_searcher.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )

        # Convert core SearchResult objects into API SearchResultItem models
        return SearchResponse(
            results=[
                SearchResultItem(
                    chunk_id=r.chunk_id,
                    content=r.chunk_content,
                    similarity_score=r.similarity_score,
                    document=DocumentInfo(
                        id=r.document_id,
                        filename=r.document_filename,
                    ),
                    position=ChunkPosition(
                        chunk_index=r.chunk_index,
                        start_char=r.start_char,
                        end_char=r.end_char,
                    ),
                    metadata=r.metadata,
                )
                for r in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keyword", response_model=KeywordSearchResponse)
async def keyword_search(
    request: KeywordSearchRequest,
    rag_context: RAGContext = Depends(get_rag_context_with_database)
):
    """
    Search using keywords with FTS5 full-text search.

    Supports:
    - Simple queries: "machine learning"
    - Boolean: "machine AND learning", "machine OR learning"
    - NOT operator: "machine NOT cat"
    - Phrase queries: '"machine learning"' (exact phrase)
    - Prefix: "mach*"

    - **query**: Search query with FTS5 syntax
    - **top_k**: Number of results (1-100)
    - **exclude_terms**: Additional terms to exclude
    - **highlight**: Highlight matched terms with <mark> tags
    """
    try:
        # Perform FTS5 search
        raw_results = rag_context.safe_keyword_searcher.search(
            query=request.query,
            top_k=request.top_k,
            exclude_terms=request.exclude_terms,
            exact_match=request.exact_match
        )

        # Enrich results with document context
        enriched_results = []
        for chunk_id, score in raw_results:
            # Get chunk from document store
            chunk = rag_context.safe_document_store.get_chunk(chunk_id)
            if not chunk:
                continue

            # Get document
            document = rag_context.safe_document_store.get_document(chunk['document_id'])
            if not document:
                continue

            # Get content (highlighted if requested)
            if request.highlight:
                content = rag_context.safe_keyword_searcher.highlight(chunk_id, request.query)
                if not content:  # Fallback if highlight fails
                    content = chunk['content']
            else:
                content = chunk['content']

            enriched_results.append(KeywordSearchResultItem(
                chunk_id=chunk_id,
                content=content,
                score=score,
                document=DocumentInfo(
                    id=document['id'],
                    filename=document['filename']
                ),
                position=ChunkPosition(
                    chunk_index=chunk['chunk_index'],
                    start_char=chunk['start_char'],
                    end_char=chunk['end_char']
                ),
                metadata=chunk['metadata']
            ))

        return KeywordSearchResponse(
            results=enriched_results,
            query=request.query
        )

    except ValueError as e:
        # FTS5 query syntax error
        raise HTTPException(status_code=400, detail=f"Invalid query syntax: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/keyword/advanced", response_model=KeywordSearchResponse)
async def advanced_keyword_search(
    request: AdvancedKeywordSearchRequest,
    rag_context: RAGContext = Depends(get_rag_context_with_database)
):
    """
    Advanced keyword search with explicit control over required/optional/excluded terms.

    - **required_keywords**: All must be present (AND)
    - **optional_keywords**: At least one should be present (OR)
    - **exclude_keywords**: None should be present (NOT)
    - **top_k**: Number of results (1-100)
    - **highlight**: Highlight matched terms
    """
    try:
        # Perform advanced FTS5 search
        raw_results = rag_context.safe_keyword_searcher.search_with_keywords(
            required_keywords=request.required_keywords,
            optional_keywords=request.optional_keywords,
            exclude_keywords=request.exclude_keywords,
            top_k=request.top_k
        )

        # Build query string for highlighting
        query_parts = []
        if request.required_keywords:
            query_parts.extend(request.required_keywords)
        if request.optional_keywords:
            query_parts.extend(request.optional_keywords)
        highlight_query = " OR ".join(query_parts)

        # Enrich results
        enriched_results = []
        for chunk_id, score in raw_results:
            chunk = rag_context.safe_document_store.get_chunk(chunk_id)
            if not chunk:
                continue

            document = rag_context.safe_document_store.get_document(chunk['document_id'])
            if not document:
                continue

            # Highlight if requested
            if request.highlight and highlight_query:
                content = rag_context.safe_keyword_searcher.highlight(chunk_id, highlight_query)
                if not content:
                    content = chunk['content']
            else:
                content = chunk['content']

            enriched_results.append(KeywordSearchResultItem(
                chunk_id=chunk_id,
                content=content,
                score=score,
                document=DocumentInfo(
                    id=document['id'],
                    filename=document['filename']
                ),
                position=ChunkPosition(
                    chunk_index=chunk['chunk_index'],
                    start_char=chunk['start_char'],
                    end_char=chunk['end_char']
                ),
                metadata=chunk['metadata']
            ))

        # Build query description
        query_desc = f"Required: {', '.join(request.required_keywords)}"
        if request.optional_keywords:
            query_desc += f" | Optional: {', '.join(request.optional_keywords)}"
        if request.exclude_keywords:
            query_desc += f" | Exclude: {', '.join(request.exclude_keywords)}"

        return KeywordSearchResponse(
            results=enriched_results,
            query=query_desc
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid query: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
