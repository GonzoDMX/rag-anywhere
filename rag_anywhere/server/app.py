# rag_anywhere/server/app.py
import argparse
from fastapi import FastAPI
import uvicorn

from .lifecycle import lifespan
from .routes import search, documents, admin

# Create FastAPI app
app = FastAPI(
    title="RAG Anywhere",
    description="Secure, portable, local-first RAG system",
    version="0.1.0",
    lifespan=lifespan
)

# Include routers
app.include_router(search.router)
app.include_router(documents.router)
app.include_router(admin.router)

# Legacy endpoint for backwards compatibility
@app.get("/status")
async def root_status():
    """Root status endpoint (redirects to admin status)"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/admin/status")


def main():
    """Main entry point for running the server"""
    import sys
    import os

    # Log to stderr immediately (before any other initialization)
    sys.stderr.write("=== RAG Anywhere Server Starting ===\n")
    sys.stderr.flush()

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--port', type=int, default=8000)
        parser.add_argument('--db', type=str, required=True)
        args = parser.parse_args()

        sys.stderr.write(f"Arguments parsed: port={args.port}, db={args.db}\n")
        sys.stderr.flush()

        # Store in app state
        app.state.port = args.port
        app.state.db_name = args.db

        # Determine log level from environment or default to info
        log_level = os.environ.get('RAG_ANYWHERE_LOG_LEVEL', 'info')
        if os.environ.get('RAG_ANYWHERE_DEBUG') == '1':
            log_level = 'debug'

        sys.stderr.write(f"Starting uvicorn on port {args.port} with log_level={log_level}...\n")
        sys.stderr.flush()

        # Run server
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=args.port,
            log_level=log_level  # Changed from "warning" to allow startup errors
        )

    except Exception as e:
        sys.stderr.write(f"FATAL ERROR in main(): {e}\n")
        import traceback
        sys.stderr.write(traceback.format_exc())
        sys.stderr.flush()
        sys.exit(1)


if __name__ == "__main__":
    main()

