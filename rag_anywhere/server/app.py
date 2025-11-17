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
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--db', type=str, required=True)
    args = parser.parse_args()
    
    # Store in app state
    app.state.port = args.port
    app.state.db_name = args.db
    
    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_level="warning"
    )


if __name__ == "__main__":
    main()

