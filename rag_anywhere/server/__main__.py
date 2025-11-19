"""
Module entry point for running the RAG Anywhere server.

This allows the server to be started with:
    python -m rag_anywhere.server
"""

from .app import main

if __name__ == "__main__":
    main()
