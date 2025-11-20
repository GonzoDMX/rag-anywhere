# rag_anywhere/server/routes/documents.py

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException

from ..models import (
    AddDocumentRequest, AddDocumentResponse,
    RemoveDocumentRequest, RemoveDocumentResponse,
    ListDocumentsResponse, DocumentListItem,
    BatchAddRequest, BatchAddResponse, BatchDocumentResult, BatchAddSummary
)
from ..dependencies import get_rag_context_with_database
from ...cli.context import RAGContext
from ...core.splitters import SplitterFactory

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/add", response_model=AddDocumentResponse)
async def add_document(
    request: AddDocumentRequest,
    rag_context: RAGContext = Depends(get_rag_context_with_database)
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
                token_estimator=rag_context.safe_embedding_provider.estimate_tokens,
                **{k: v for k, v in splitter_config.items() if k != 'strategy'}
            )
            
            original_splitter = rag_context.safe_indexer.splitter
            rag_context.safe_indexer.splitter = splitter
        
        # Index document
        doc_id = rag_context.safe_indexer.index_document(file_path, request.metadata)
        
        # Restore original splitter
        if original_splitter:
            rag_context.safe_indexer.splitter = original_splitter
        
        return AddDocumentResponse(
            status="success",
            document_id=doc_id,
            filename=file_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/add-batch", response_model=BatchAddResponse)
async def add_documents_batch(
    request: BatchAddRequest,
    rag_context: RAGContext = Depends(get_rag_context_with_database)
):
    """
    Add multiple documents in a single API call.

    - **documents**: List of documents to add (file_path, metadata, splitter_overrides)
    - **fail_fast**: If true, stop processing on first error

    Returns detailed results for each document and a summary.
    """
    results = []
    succeeded = 0
    failed = 0

    for doc_item in request.documents:
        try:
            file_path = Path(doc_item.file_path)

            if not file_path.exists():
                results.append(BatchDocumentResult(
                    file_path=doc_item.file_path,
                    status="error",
                    error=f"File not found: {file_path}"
                ))
                failed += 1

                if request.fail_fast:
                    break
                continue

            # Apply splitter overrides if provided
            original_splitter = None
            if doc_item.splitter_overrides:
                file_ext = file_path.suffix.lower()
                splitter_config = rag_context.get_splitter_config(file_ext, doc_item.splitter_overrides)

                splitter = SplitterFactory.create_splitter(
                    splitter_config['strategy'],
                    token_estimator=rag_context.safe_embedding_provider.estimate_tokens,
                    **{k: v for k, v in splitter_config.items() if k != 'strategy'}
                )

                original_splitter = rag_context.safe_indexer.splitter
                rag_context.safe_indexer.splitter = splitter

            # Index document
            doc_id = rag_context.safe_indexer.index_document(file_path, doc_item.metadata)

            # Restore original splitter
            if original_splitter:
                rag_context.safe_indexer.splitter = original_splitter

            results.append(BatchDocumentResult(
                file_path=doc_item.file_path,
                status="success",
                document_id=doc_id,
                filename=file_path.name
            ))
            succeeded += 1

        except Exception as e:
            results.append(BatchDocumentResult(
                file_path=doc_item.file_path,
                status="error",
                error=str(e)
            ))
            failed += 1

            if request.fail_fast:
                break

    # Determine overall status
    status = "completed" if failed == 0 else "partial"

    return BatchAddResponse(
        status=status,
        results=results,
        summary=BatchAddSummary(
            total=len(request.documents),
            succeeded=succeeded,
            failed=failed
        )
    )


@router.post("/remove", response_model=RemoveDocumentResponse)
async def remove_document(
    request: RemoveDocumentRequest,
    rag_context: RAGContext = Depends(get_rag_context_with_database)
):
    """
    Remove a document from the database.
    
    - **document_id**: UUID of the document to remove
    """
    try:
        success = rag_context.safe_indexer.remove_document(request.document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return RemoveDocumentResponse(status="success")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=ListDocumentsResponse)
async def list_documents(
    rag_context: RAGContext = Depends(get_rag_context_with_database)
):
    """
    List all documents in the database.
    
    Returns a list of documents with metadata and chunk counts.
    """
    try:
        documents = rag_context.safe_document_store.list_documents()
        
        # Add chunk count for each document
        doc_items = []
        for doc in documents:
            chunks = rag_context.safe_document_store.get_chunks_by_document(doc['id'])
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
    rag_context: RAGContext = Depends(get_rag_context_with_database)
):
    """
    Get detailed information about a specific document.
    
    - **document_id**: UUID of the document
    """
    try:
        document = rag_context.safe_document_store.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Add chunks
        chunks = rag_context.safe_document_store.get_chunks_by_document(document_id)
        document['chunks'] = chunks
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
