# rag_anywhere/server/dependencies.py
from fastapi import HTTPException

from .lifecycle import lifecycle


def get_rag_context():
    """Dependency to get RAG context"""
    if not lifecycle.rag_context:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Update activity timestamp
    if lifecycle.server_state:
        lifecycle.server_state.update_activity()
    
    return lifecycle.rag_context


def get_rag_context_with_database():
    """
    Get RAG context and ensure a database is loaded.
    Raises HTTP 503 if no database is active.
    """
    rag_context = get_rag_context()
    
    if rag_context.active_db_name is None:
        raise HTTPException(
            status_code=503,
            detail="No database is currently loaded. Load a database first using the /admin/load-database endpoint."
        )
    
    # These should never be None if active_db_name is set, but double-check
    if (rag_context.indexer is None or 
        rag_context.embedding_provider is None or
        rag_context.document_store is None or
        rag_context.searcher is None or
        rag_context.keyword_searcher is None):
        raise HTTPException(
            status_code=503,
            detail="Database components not fully initialized. Try reloading the database."
        )
    
    return rag_context


def get_server_state():
    """Dependency to get server state"""
    if not lifecycle.server_state:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return lifecycle.server_state
