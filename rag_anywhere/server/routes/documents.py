# rag_anywhere/server/routes/documents.py
from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path

from ..models import (
    AddDocumentRequest, AddDocumentResponse,
    RemoveDocumentRequest, RemoveDocumentResponse,
    ListDocumentsResponse, DocumentListItem
)
from ..dependencies import get_rag_context
from ...cli.context import RAGContext
from ...core.splitters import SplitterFactory

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/add", response_model=AddDocumentResponse)
async def add_document(
    request: AddDocumentRequest,
    rag_context: RAGContext = Depends(get_rag_context)
):
    """
    Add a document to the database.
    
    - **file_path**: Absolute path to the document file
    - **metadata**: Optional metadata dictionary
    - **splitter_overrides**: Optional splitter configuration
    """
    try:
        file_path = Path(request.file_path)
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Apply splitter overrides if provided
        original_splitter = None
        if request.splitter_overrides:
            file_ext = file_path.suffix.lower()
            splitter_config = rag_context.get_splitter_config(file_ext, request.splitter_overrides)
            
            splitter = SplitterFactory.create_splitter(
                splitter_config['strategy'],
                token_estimator=rag_context.embedding_provider.estimate_tokens,
                **{k: v for k, v in splitter_config.items() if k != 'strategy'}
            )
            
            original_splitter = rag_context.indexer.splitter
            rag_context.indexer.splitter = splitter
        
        # Index document
        doc_id = rag_context.indexer.index_document(file_path, request.metadata)
        
        # Restore original splitter
        if original_splitter:
            rag_context.indexer.splitter = original_splitter
        
        return AddDocumentResponse(
            status="success",
            document_id=doc_id,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/remove", response_model=RemoveDocumentResponse)
async def remove_document(
    request: RemoveDocumentRequest,
    rag_context: RAGContext = Depends(get_rag_context)
):
    """
    Remove a document from the database.
    
    - **document_id**: UUID of the document to remove
    """
    try:
        success = rag_context.indexer.remove_document(request.document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return RemoveDocumentResponse(status="success")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=ListDocumentsResponse)
async def list_documents(
    rag_context: RAGContext = Depends(get_rag_context)
):
    """
    List all documents in the database.
    
    Returns a list of documents with metadata and chunk counts.
    """
    try:
        documents = rag_context.document_store.list_documents()
        
        # Add chunk count for each document
        doc_items = []
        for doc in documents:
            chunks = rag_context.document_store.get_chunks_by_document(doc['id'])
            doc_items.append(DocumentListItem(
                id=doc['id'],
                filename=doc['filename'],
                created_at=doc['created_at'],
                metadata=doc['metadata'],
                num_chunks=len(chunks)
            ))
        
        return ListDocumentsResponse(documents=doc_items)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}")
async def get_document(
    document_id: str,
    rag_context: RAGContext = Depends(get_rag_context)
):
    """
    Get detailed information about a specific document.
    
    - **document_id**: UUID of the document
    """
    try:
        document = rag_context.document_store.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Add chunks
        chunks = rag_context.document_store.get_chunks_by_document(document_id)
        document['chunks'] = chunks
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
