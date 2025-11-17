# rag_anywhere/server/dependencies.py
from fastapi import HTTPException

from .lifecycle import lifecycle
from .state import ServerState


def get_rag_context():
    """Dependency to get RAG context"""
    if not lifecycle.rag_context:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Update activity timestamp
    if lifecycle.server_state:
        lifecycle.server_state.update_activity()
    
    return lifecycle.rag_context


def get_server_state():
    """Dependency to get server state"""
    if not lifecycle.server_state:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return lifecycle.server_state
