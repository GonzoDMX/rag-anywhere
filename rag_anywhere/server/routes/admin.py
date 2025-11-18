# rag_anywhere/server/routes/admin.py
from fastapi import APIRouter, Depends, HTTPException

from ..models import StatusResponse, ReloadRequest, ReloadResponse, SleepResponse
from ..dependencies import get_rag_context, get_server_state
from ...cli.context import RAGContext
from ...server.state import ServerState, ServerStatus

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/status", response_model=StatusResponse)
async def status(rag_context: RAGContext = Depends(get_rag_context)):
    """
    Get current server status.
    
    Returns information about the active database and content statistics.
    """
    # No active DB: report zeros instead of erroring
    if (
        rag_context.active_db_name is None
        or rag_context.document_store is None
        or rag_context.vector_store is None
    ):
        return StatusResponse(
            status="running",
            active_database=rag_context.active_db_name,
            num_documents=0,
            num_vectors=0,
        )

    # Active DB and stores available: compute real stats
    return StatusResponse(
        status="running",
        active_database=rag_context.active_db_name,
        num_documents=len(rag_context.document_store.list_documents()),
        num_vectors=rag_context.vector_store.count(),
    )


@router.post("/reload", response_model=ReloadResponse)
async def reload(
    request: ReloadRequest,
    rag_context: RAGContext = Depends(get_rag_context)
):
    """
    Reload the database (and optionally the embedding model).
    
    - **database**: Name of the database to load
    - **reload_model**: Whether to reload the embedding model (if changed)
    """
    try:
        if request.reload_model:
            print(f"Reloading with new embedding model for database '{request.database}'")
        else:
            print(f"Reloading database '{request.database}' (keeping existing model)")
        
        # Reload database
        rag_context.load_database(request.database, verbose=True)
        
        return ReloadResponse(
            status="success",
            database=request.database
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sleep", response_model=SleepResponse)
async def sleep_server(state: ServerState = Depends(get_server_state)):
    """
    Put the server into sleep mode.
    
    The server will remain running but inactive until the next request.
    """
    state.update_status(ServerStatus.SLEEPING)
    return SleepResponse(status="sleeping")
