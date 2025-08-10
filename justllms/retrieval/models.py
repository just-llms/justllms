"""Data models for retrieval functionality."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Represents a document in the retrieval system."""
    
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique document ID")
    content: str = Field(..., description="The document content/text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    embedding: Optional[List[float]] = Field(None, description="Document embedding vector")
    score: Optional[float] = Field(None, description="Relevance score (for search results)")
    
    class Config:
        arbitrary_types_allowed = True


class RetrievalResult(BaseModel):
    """Result from a retrieval operation."""
    
    documents: List[Document] = Field(..., description="Retrieved documents")
    query: str = Field(..., description="The search query")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RetrievalConfig(BaseModel):
    """Configuration for retrieval system."""
    
    # Embedding configuration
    embedding_provider: Optional[str] = Field(None, description="Embedding provider (e.g., 'openai/text-embedding-3-large')")
    embedding_model: Optional[str] = Field(None, description="Specific embedding model name")
    embedding_config: Optional[Dict[str, Any]] = Field(None, description="Additional embedding provider configuration")
    
    # Vector store configuration  
    vector_store: Dict[str, Any] = Field(..., description="Vector store configuration")
    
    # Document processing
    chunk_size: int = Field(512, description="Size of text chunks for processing")
    chunk_overlap: int = Field(50, description="Overlap between chunks")
    splitting_strategy: str = Field("recursive", description="Text splitting strategy")
    
    # Retrieval settings
    default_k: int = Field(5, description="Default number of results to retrieve")
    similarity_threshold: float = Field(0.0, description="Minimum similarity threshold")
    
    # Processing options
    extract_metadata: bool = Field(True, description="Extract metadata from documents")
    clean_text: bool = Field(True, description="Clean and normalize text")
    filter_languages: Optional[List[str]] = Field(None, description="Allowed languages")


class CollectionInfo(BaseModel):
    """Information about a document collection."""
    
    name: str = Field(..., description="Collection name")
    document_count: int = Field(0, description="Number of documents")
    vector_count: int = Field(0, description="Number of vectors") 
    created_at: str = Field(..., description="Creation timestamp")
    updated_at: str = Field(..., description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Collection metadata")
    
    
class ProcessingResult(BaseModel):
    """Result from document processing operation."""
    
    documents_processed: int = Field(..., description="Number of documents processed")
    chunks_created: int = Field(..., description="Number of chunks created")
    vectors_generated: int = Field(..., description="Number of vectors generated")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    errors: List[str] = Field(default_factory=list, description="Processing errors")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")