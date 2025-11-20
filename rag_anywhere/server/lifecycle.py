# rag_anywhere/server/lifecycle.py

import sys
import signal
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from ..config import Config
from ..cli.context import RAGContext
from ..utils.logging import get_logger
from .state import ServerState

logger = get_logger('server.lifecycle')


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

        logger.info(f"Loading database '{db_name}'...")
        self.rag_context.load_database(db_name, verbose=True)

        logger.info(f"✓ Server ready on port {port}")
    
    def shutdown(self):
        """Cleanup server resources"""
        logger.info("Shutting down server...")

        # Cleanup resources if needed
        if self.rag_context:
            # Clear references to allow garbage collection
            self.rag_context.embedding_provider = None
            self.rag_context.vector_store = None
            self.rag_context.document_store = None

        logger.info("✓ Server shut down")


# Global lifecycle instance
lifecycle = ServerLifecycle()


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context manager"""
    try:
        # Startup
        logger.info(f"Starting server lifecycle for database '{app.state.db_name}' on port {app.state.port}")
        sys.stderr.write(f"Lifecycle: Starting setup for {app.state.db_name}...\n")
        sys.stderr.flush()

        lifecycle.setup(app.state.db_name, app.state.port)

        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("Received shutdown signal")
            sys.stderr.write("\nReceived shutdown signal...\n")
            sys.stderr.flush()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Server lifecycle startup completed successfully")
        sys.stderr.write("Lifecycle: Setup complete\n")
        sys.stderr.flush()

        yield

    except Exception as e:
        # Log to both logger and stderr for maximum visibility
        error_msg = f"FATAL: Server startup failed during lifespan: {e}"
        logger.error(error_msg, exc_info=True)

        # Write directly to stderr (bypasses buffering issues)
        sys.stderr.write(f"\n{'='*60}\n")
        sys.stderr.write(f"{error_msg}\n")
        sys.stderr.write(f"{'='*60}\n")
        sys.stderr.write(traceback.format_exc())
        sys.stderr.write(f"{'='*60}\n\n")
        sys.stderr.flush()

        raise

    finally:
        # Shutdown - always runs even if startup failed
        try:
            logger.info("Starting server shutdown")
            lifecycle.shutdown()
            logger.info("Server shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}", exc_info=True)
            sys.stderr.write(f"Error during shutdown: {e}\n")
            sys.stderr.write(traceback.format_exc())
            sys.stderr.flush()
