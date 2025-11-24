# rag_anywhere/cli/main.py

import os
import typer

from .commands import db, documents, search, info, server, kg
from ..config.settings import Config
from ..utils.logging import setup_logging

app = typer.Typer(
    name="rag-anywhere",
    help="Secure, portable, local-first RAG system",
    add_completion=False
)

# Add command groups
app.add_typer(db.app, name="db", help="Database management")
app.add_typer(server.app, name="server", help="Server management")
app.add_typer(kg.app, name="kg", help="Knowledge graph operations")

# Add direct document commands at root level
app.command(name="add")(documents.add)
app.command(name="remove")(documents.remove)
app.command(name="list")(documents.list_documents)

# Add search as a command group
app.add_typer(search.app, name="search", help="Search for documents")

# Add status at root level
app.command(name="status")(info.show_status)


@app.callback()
def callback(
    verbose: bool = typer.Option(
        False,
        "-v",
        "--verbose",
        help="Enable verbose output with debug logging"
    )
):
    """
    RAG Anywhere - Secure, portable, local-first RAG system
    """
    # Store verbose state in environment variable so all commands can access it
    if verbose:
        os.environ['RAG_ANYWHERE_DEBUG'] = '1'

    # Initialize logging globally
    config = Config()
    setup_logging(config.config_dir, debug=verbose)


def main():
    app()


if __name__ == "__main__":
    main()
