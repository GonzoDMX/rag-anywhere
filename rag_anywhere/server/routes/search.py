# rag_anywhere/server/routes/search.py
from fastapi import APIRouter, Depends, HTTPException

from ..models import SearchRequest, SearchResponse
from ..dependencies import get_rag_context
from ...cli.context import RAGContext

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    rag_context: RAGContext = Depends(get_rag_context)
):
    """
    Search for documents using semantic similarity.
    
    - **query**: The search query text
    - **top_k**: Number of results to return (1-100)
    - **min_score**: Minimum similarity score threshold (0.0-1.0)
    """
    try:
        results = rag_context.searcher.search(
            query=request.query,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        return SearchResponse(
            results=[r.to_dict() for r in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
