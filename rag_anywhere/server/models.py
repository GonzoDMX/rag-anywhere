# rag_anywhere/server/models.py

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Any


# Request Models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    top_k: int = Field(5, ge=1, le=100, description="Number of results to return")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    task: str = Field("retrieval", description="Task type for embedding: retrieval, fact_checking, code_retrieval, etc.")


class KeywordSearchRequest(BaseModel):
    """
    Unified keyword search request supporting both free-form and structured modes.

    Free-form mode: Provide 'query' for FTS5 syntax (AND, OR, NOT, phrases, prefix)
    Structured mode: Provide 'required_keywords' and/or 'optional_keywords'
    """
    # Free-form mode fields
    query: Optional[str] = Field(None, description="Keyword search query (supports AND, OR, NOT, phrases, prefix)")
    exclude_terms: Optional[List[str]] = Field(None, description="Terms to exclude from results (free-form mode)")
    exact_match: bool = Field(False, description="Treat query as exact phrase match (free-form mode)")

    # Structured mode fields
    required_keywords: Optional[List[str]] = Field(None, description="All these terms must be present (AND)")
    optional_keywords: Optional[List[str]] = Field(None, description="At least one should be present (OR)")
    exclude_keywords: Optional[List[str]] = Field(None, description="None of these should be present (NOT)")

    # Common fields
    top_k: int = Field(10, ge=1, le=100, description="Number of results to return")
    highlight: bool = Field(False, description="Highlight matched terms in results")

    @model_validator(mode='after')
    def validate_search_mode(self):
        """Ensure exactly one search mode is specified"""
        has_freeform = self.query is not None
        has_structured = self.required_keywords is not None or self.optional_keywords is not None

        if not has_freeform and not has_structured:
            raise ValueError("Must provide either 'query' (free-form mode) or 'required_keywords'/'optional_keywords' (structured mode)")

        if has_freeform and has_structured:
            raise ValueError("Cannot mix free-form mode ('query') with structured mode ('required_keywords'/'optional_keywords')")

        # Validate structured mode has at least some keywords
        if has_structured and not self.required_keywords and not self.optional_keywords:
            raise ValueError("Structured mode requires at least 'required_keywords' or 'optional_keywords'")

        # Validate free-form mode incompatible fields
        if has_freeform:
            if self.exclude_keywords is not None:
                raise ValueError("Use 'exclude_terms' instead of 'exclude_keywords' in free-form mode")

        # Validate structured mode incompatible fields
        if has_structured:
            if self.exclude_terms is not None:
                raise ValueError("Use 'exclude_keywords' instead of 'exclude_terms' in structured mode")
            if self.exact_match:
                raise ValueError("'exact_match' is only supported in free-form mode")

        return self


class AddDocumentRequest(BaseModel):
    file_path: str = Field(..., description="Absolute path to the document file")
    doc_type: str = Field("text", description="Document type: 'text' or 'code'")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the document")
    splitter_overrides: Optional[Dict[str, Any]] = Field(None, description="Optional splitter configuration overrides")


class RemoveDocumentRequest(BaseModel):
    document_id: str = Field(..., description="UUID of the document to remove")


class BatchDocumentItem(BaseModel):
    file_path: str = Field(..., description="Absolute path to the document file")
    doc_type: str = Field("text", description="Document type: 'text' or 'code'")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata for the document")
    splitter_overrides: Optional[Dict[str, Any]] = Field(None, description="Optional splitter configuration overrides")


class BatchAddRequest(BaseModel):
    documents: List[BatchDocumentItem] = Field(..., description="List of documents to add")
    fail_fast: bool = Field(False, description="Stop processing on first error")


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
    doc_type: str
    created_at: str
    metadata: Dict[str, Any]
    num_chunks: int


class ListDocumentsResponse(BaseModel):
    documents: List[DocumentListItem]


# Knowledge Graph Models
class EntityItem(BaseModel):
    id: int
    name: str
    display_name: str
    category: str
    frequency: int


class EntityListResponse(BaseModel):
    entities: List[EntityItem]
    total: int


class EntityDetailsResponse(BaseModel):
    entity: EntityItem
    chunk_ids: List[str]
    related_entities: List[Dict[str, Any]]


class KGStatsResponse(BaseModel):
    total_entities: int
    total_edges: int
    by_category: Dict[str, int]
    top_entities: List[Dict[str, Any]]


class ReprocessRequest(BaseModel):
    document_id: Optional[str] = Field(None, description="Document ID to reprocess")
    labels: Optional[List[str]] = Field(None, description="Additional entity labels")


class ReprocessResponse(BaseModel):
    status: str
    documents_processed: int
    total_entities: int


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


class BatchDocumentResult(BaseModel):
    file_path: str
    status: str  # "success" or "error"
    document_id: Optional[str] = None
    filename: Optional[str] = None
    error: Optional[str] = None


class BatchAddSummary(BaseModel):
    total: int
    succeeded: int
    failed: int


class BatchAddResponse(BaseModel):
    status: str  # "completed" or "partial"
    results: List[BatchDocumentResult]
    summary: BatchAddSummary


class KeywordSearchResultItem(BaseModel):
    chunk_id: str
    content: str  # Highlighted if requested
    score: float
    document: DocumentInfo
    position: ChunkPosition
    metadata: Dict[str, Any]


class KeywordSearchResponse(BaseModel):
    results: List[KeywordSearchResultItem]
    query: str
