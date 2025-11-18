# rag_anywhere/server/lifecycle.py
import signal
import sys
from contextlib import asynccontextmanager
from typing import Optional

from ..config import Config
from ..cli.context import RAGContext
from .state import ServerState


class ServerLifecycle:
    """Manages server startup and shutdown"""
    
    def __init__(self):
        self.rag_context: Optional[RAGContext] = None
        self.config: Optional[Config] = None
        self.server_state: Optional[ServerState] = None
        self.db_name: Optional[str] = None
        self.port: Optional[int] = None
    
    def setup(self, db_name: str, port: int):
        """Setup server resources"""
        self.db_name = db_name
        self.port = port
        
        # Initialize config and state
        self.config = Config()
        self.server_state = ServerState(self.config.config_dir)
        
        # Initialize RAG context
        self.rag_context = RAGContext(self.config.config_dir)
        
        print(f"Loading database '{db_name}'...")
        self.rag_context.load_database(db_name, verbose=True)
        
        print(f"✓ Server ready on port {port}")
    
    def shutdown(self):
        """Cleanup server resources"""
        print("Shutting down server...")
        
        # Cleanup resources if needed
        if self.rag_context:
            # Clear references to allow garbage collection
            self.rag_context.embedding_provider = None
            self.rag_context.vector_store = None
            self.rag_context.document_store = None
        
        print("✓ Server shut down")


# Global lifecycle instance
lifecycle = ServerLifecycle()


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager"""
    # Startup
    lifecycle.setup(app.state.db_name, app.state.port)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        print("\nReceived shutdown signal...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    yield
    
    # Shutdown
    lifecycle.shutdown()
