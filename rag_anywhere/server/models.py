# rag_anywhere/server/models.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# Request Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")


class AddDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the document file")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the document")
    splitter_overrides: Optional[Dict[str, Any]] = Field(None, description="Optional splitter configuration overrides")


class RemoveDocumentRequest(BaseModel):
    document_id: str = Field(..., description="UUID of the document to remove")


class ReloadRequest(BaseModel):
    database: str = Field(..., description="Database name to reload")
    reload_model: bool = Field(False, description="Whether to reload the embedding model")


# Response Models
class DocumentInfo(BaseModel):
    id: str
    filename: str


class ChunkPosition(BaseModel):
    chunk_index: int
    start_char: int
    end_char: int


class SearchResultItem(BaseModel):
    chunk_id: str
    content: str
    similarity_score: float
    document: DocumentInfo
    position: ChunkPosition
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class AddDocumentResponse(BaseModel):
    status: str
    document_id: str
    filename: str


class RemoveDocumentResponse(BaseModel):
    status: str


class DocumentListItem(BaseModel):
    id: str
    filename: str
    created_at: str
    metadata: Dict[str, Any]
    num_chunks: int


class ListDocumentsResponse(BaseModel):
    documents: List[DocumentListItem]


class StatusResponse(BaseModel):
    status: str
    active_database: Optional[str]
    num_documents: int
    num_vectors: int


class ReloadResponse(BaseModel):
    status: str
    database: str


class SleepResponse(BaseModel):
    status: str
