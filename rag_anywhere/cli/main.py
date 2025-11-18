# rag_anywhere/cli/main.py
import typer

from .commands import db, documents, search, info, server

app = typer.Typer(
    name="rag-anywhere",
    help="Secure, portable, local-first RAG system",
    add_completion=False
)

# Add command groups
app.add_typer(db.app, name="db", help="Database management")
app.add_typer(server.app, name="server", help="Server management")
app.add_typer(documents.app, name="doc", help="Document management (alternative to direct commands)")

# Add direct document commands at root level
app.command(name="add")(documents.add)
app.command(name="remove")(documents.remove)
app.command(name="list")(documents.list_documents)

# Add search as a command group
app.add_typer(search.app, name="search", help="Search for documents")

# Add info at root level
app.command(name="info")(info.show_info)
app.command(name="status")(info.show_status)


@app.callback()
def callback():
    """
    RAG Anywhere - Secure, portable, local-first RAG system
    """
    pass


def main():
    app()


if __name__ == "__main__":
    main()
