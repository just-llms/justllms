"""Retrieval-Augmented Generation (RAG) functionality for JustLLMs."""

from justllms.retrieval.embeddings import EmbeddingProvider
from justllms.retrieval.models import Document, RetrievalConfig, RetrievalResult
from justllms.retrieval.retrievers import RetrievalManager
from justllms.retrieval.vector_stores import VectorStore

__all__ = [
    "EmbeddingProvider",
    "VectorStore", 
    "RetrievalManager",
    "Document",
    "RetrievalConfig",
    "RetrievalResult",
]