"""Vector store implementations for RAG functionality."""

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from justllms.retrieval.models import CollectionInfo, Document, RetrievalResult


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def search(
        self, 
        query_vector: List[float], 
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def asearch(
        self,
        query_vector: List[float],
        collection: str, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents (async)."""
        pass
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        collection: str,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to the store."""
        pass
    
    @abstractmethod
    async def aadd_documents(
        self,
        documents: List[Document], 
        collection: str,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to the store (async)."""
        pass
    
    @abstractmethod
    def delete_documents(
        self,
        document_ids: List[str],
        collection: str,
        **kwargs: Any
    ) -> bool:
        """Delete documents from the store."""
        pass
    
    @abstractmethod
    def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any
    ) -> bool:
        """Create a new collection."""
        pass
    
    @abstractmethod
    def list_collections(self) -> List[CollectionInfo]:
        """List all collections."""
        pass
    
    @abstractmethod
    def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get collection information."""
        pass
    
    def search_with_query(
        self,
        query: str,
        embedding_provider: Any,
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> RetrievalResult:
        """Search using a text query (convenience method)."""
        start_time = time.time()
        
        # Generate embedding for query
        query_vector = embedding_provider.embed_text(query)
        
        # Search for documents
        documents = self.search(
            query_vector=query_vector,
            collection=collection,
            k=k,
            filters=filters,
            **kwargs
        )
        
        search_time_ms = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            documents=documents,
            query=query,
            total_results=len(documents),
            search_time_ms=search_time_ms
        )
    
    def search_text(
        self,
        query: str,
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> RetrievalResult:
        """Search using text without embeddings (fallback method)."""
        # This is a fallback when no embedding provider is available
        # For now, return empty results as most vector stores require embeddings
        return RetrievalResult(
            documents=[],
            query=query,
            total_results=0,
            search_time_ms=0.0
        )


class ChromaVectorStore(VectorStore):
    """Chroma vector store implementation."""
    
    def __init__(self, host: str = "localhost", port: int = 8000, **kwargs: Any):
        self.host = host
        self.port = port
        self._client = None
        
    def _get_client(self):
        """Lazy load Chroma client."""
        if self._client is None:
            try:
                import chromadb
                if self.host == "localhost":
                    self._client = chromadb.Client()
                else:
                    self._client = chromadb.HttpClient(host=self.host, port=self.port)
            except ImportError as e:
                raise ImportError("ChromaDB package not installed. Install with: pip install chromadb") from e
        return self._client
    
    def search(
        self,
        query_vector: List[float],
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents."""
        client = self._get_client()
        collection_obj = client.get_collection(name=collection)
        
        results = collection_obj.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=filters,
            **kwargs
        )
        
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, doc_content in enumerate(results["documents"][0]):
                doc_id = results["ids"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else None
                
                # Convert distance to similarity score (Chroma uses L2 distance)
                score = 1 / (1 + distance) if distance is not None else None
                
                documents.append(Document(
                    id=doc_id,
                    content=doc_content,
                    metadata=metadata,
                    score=score
                ))
        
        return documents
    
    def search_text(
        self,
        query: str,
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> RetrievalResult:
        """Search using text query with ChromaDB's built-in embeddings."""
        import time
        start_time = time.time()
        
        try:
            client = self._get_client()
            collection_obj = client.get_collection(name=collection)
            
            # Use ChromaDB's built-in query method
            results = collection_obj.query(
                query_texts=[query],
                n_results=k,
                where=filters,
                **kwargs
            )
            
            # Convert ChromaDB results to our Document format
            documents = []
            if results["documents"] and results["documents"][0]:
                for i, doc_content in enumerate(results["documents"][0]):
                    doc_id = results["ids"][0][i]
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    distance = results["distances"][0][i] if results["distances"] else None
                    
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    score = 1 / (1 + distance) if distance is not None else None
                    
                    documents.append(Document(
                        id=doc_id,
                        content=doc_content,
                        metadata=metadata or {},
                        score=score
                    ))
            
            search_time_ms = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                documents=documents,
                query=query,
                total_results=len(documents),
                search_time_ms=search_time_ms
            )
            
        except Exception:
            # Fallback to empty results if search fails
            search_time_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                documents=[],
                query=query,
                total_results=0,
                search_time_ms=search_time_ms
            )
    
    async def asearch(
        self,
        query_vector: List[float],
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents (async)."""
        # Chroma doesn't have async client yet, so we use sync
        return self.search(query_vector, collection, k, filters, **kwargs)
    
    def add_documents(
        self,
        documents: List[Document],
        collection: str,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to the store."""
        client = self._get_client()
        
        try:
            collection_obj = client.get_collection(name=collection)
        except Exception:
            # Collection doesn't exist, create it
            collection_obj = client.create_collection(
                name=collection,
                metadata=kwargs.get("metadata", {})
            )
        
        # Prepare data for insertion
        ids = [doc.id for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        embeddings = [doc.embedding for doc in documents if doc.embedding]
        
        # ChromaDB can auto-generate embeddings if none provided
        if embeddings and len(embeddings) == len(documents):
            # Use provided embeddings
            collection_obj.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            # Let ChromaDB generate embeddings automatically
            collection_obj.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
                # No embeddings parameter - ChromaDB will generate them
            )
        
        return ids
    
    async def aadd_documents(
        self,
        documents: List[Document],
        collection: str,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to the store (async)."""
        return self.add_documents(documents, collection, **kwargs)
    
    def delete_documents(
        self,
        document_ids: List[str],
        collection: str,
        **kwargs: Any
    ) -> bool:
        """Delete documents from the store."""
        client = self._get_client()
        collection_obj = client.get_collection(name=collection)
        collection_obj.delete(ids=document_ids)
        return True
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any
    ) -> bool:
        """Create a new collection."""
        client = self._get_client()
        client.create_collection(
            name=name,
            metadata=kwargs.get("metadata", {"dimension": dimension})
        )
        return True
    
    def list_collections(self) -> List[CollectionInfo]:
        """List all collections."""
        client = self._get_client()
        collections = client.list_collections()
        
        collection_infos = []
        for collection in collections:
            count = collection.count()
            collection_infos.append(CollectionInfo(
                name=collection.name,
                document_count=count,
                vector_count=count,
                created_at="",  # Chroma doesn't provide timestamps
                updated_at="",
                metadata=collection.metadata or {}
            ))
        
        return collection_infos
    
    def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get collection information."""
        try:
            client = self._get_client()
            collection = client.get_collection(name)
            count = collection.count()
            
            return CollectionInfo(
                name=name,
                document_count=count,
                vector_count=count,
                created_at="",
                updated_at="",
                metadata=collection.metadata or {}
            )
        except Exception:
            return None


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(self, api_key: str, environment: str, index_name: str, **kwargs: Any):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self._index = None
        self._pc_client = None
    
    def _get_pc_client(self):
        """Lazy load Pinecone client."""
        if self._pc_client is None:
            try:
                from pinecone import Pinecone
                self._pc_client = Pinecone(api_key=self.api_key)
            except ImportError as e:
                raise ImportError("Pinecone package not installed. Install with: pip install pinecone") from e
        return self._pc_client
    
    def _get_index(self):
        """Lazy load Pinecone index."""
        if self._index is None:
            pc = self._get_pc_client()
            self._index = pc.Index(self.index_name)
        return self._index
    
    def search(
        self,
        query_vector: List[float],
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents."""
        index = self._get_index()
        
        # Add collection filter
        if filters is None:
            filters = {}
        filters["collection"] = collection
                
        results = index.query(
            vector=query_vector,
            top_k=k,
            filter=filters,
            include_metadata=True,
            include_values=False,
            **kwargs
        )
        
        documents = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            content = metadata.pop("content", "")
            
            documents.append(Document(
                id=match["id"],
                content=content,
                metadata=metadata,
                score=match.get("score", 0.0)
            ))
        
        return documents
    
    def search_text(
        self,
        query: str,
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> RetrievalResult:
        """Search using text query with Pinecone's built-in embeddings."""
        start_time = time.time()
        
        try:
            # Use Pinecone's inference API to generate query embedding
            pc_client = self._get_pc_client()
            query_embedding_response = pc_client.inference.embed(
                model="llama-text-embed-v2",
                inputs=[query],
                parameters={"input_type": "query"}  # Different from passage input
            )
            
            # Extract the embedding
            query_vector = query_embedding_response[0].values
            
            # Use the regular search method with the generated embedding
            documents = self.search(
                query_vector=query_vector,
                collection=collection,
                k=k,
                filters=filters,
                **kwargs
            )
            
            search_time_ms = (time.time() - start_time) * 1000
            
            return RetrievalResult(
                documents=documents,
                query=query,
                total_results=len(documents),
                search_time_ms=search_time_ms
            )
            
        except Exception:
            # Fallback to empty results if embedding generation fails
            search_time_ms = (time.time() - start_time) * 1000
            return RetrievalResult(
                documents=[],
                query=query,
                total_results=0,
                search_time_ms=search_time_ms
            )
    
    async def asearch(
        self,
        query_vector: List[float],
        collection: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Search for similar documents (async)."""
        # Pinecone client doesn't have async support, use sync
        return self.search(query_vector, collection, k, filters, **kwargs)
    
    def add_documents(
        self,
        documents: List[Document],
        collection: str,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to the store."""
        index = self._get_index()
        
        vectors_to_upsert = []
        docs_without_embeddings = []
        
        for doc in documents:
            # Add collection to metadata
            metadata = doc.metadata.copy()
            metadata["collection"] = collection
            metadata["content"] = doc.content
            
            if doc.embedding:
                # Use provided embedding
                vectors_to_upsert.append({
                    "id": doc.id,
                    "values": doc.embedding,
                    "metadata": metadata
                })
            else:
                # Store for later processing with Pinecone inference
                docs_without_embeddings.append((doc, metadata))
        
        # Handle documents with embeddings first
        if vectors_to_upsert:
            # Upsert vectors in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                index.upsert(vectors=batch)
        
        # Handle documents without embeddings using Pinecone's built-in embeddings
        if docs_without_embeddings:
            try:
                # Use Pinecone's built-in embedding generation
                pc_client = self._get_pc_client()
                
                texts_to_embed = []
                doc_metadata_pairs = []
                
                for doc, metadata in docs_without_embeddings:
                    texts_to_embed.append(doc.content)
                    doc_metadata_pairs.append((doc, metadata))
                
                # Process in batches to respect Pinecone's input limit (96 for llama-text-embed-v2)
                embedding_batch_size = 96
                all_embeddings = []
                
                for i in range(0, len(texts_to_embed), embedding_batch_size):
                    batch_texts = texts_to_embed[i:i + embedding_batch_size]
                    
                    # Use Pinecone inference API to generate embeddings for this batch
                    embeddings_response = pc_client.inference.embed(
                        model="llama-text-embed-v2",
                        inputs=batch_texts,
                        parameters={"input_type": "passage"}
                    )
                    
                    # Extract embeddings from response
                    batch_embeddings = [emb.values for emb in embeddings_response]
                    all_embeddings.extend(batch_embeddings)
                
                # Prepare vectors with generated embeddings
                pinecone_vectors = []
                for i, (doc, metadata) in enumerate(doc_metadata_pairs):
                    pinecone_vectors.append({
                        "id": doc.id,
                        "values": all_embeddings[i],
                        "metadata": metadata
                    })
                
                # Upsert vectors with generated embeddings
                if pinecone_vectors:
                    batch_size = 100
                    for i in range(0, len(pinecone_vectors), batch_size):
                        batch = pinecone_vectors[i:i + batch_size]
                        index.upsert(vectors=batch)
                        
                
            except Exception as e:
                raise e
        
        
        return [doc.id for doc in documents]
    
    async def aadd_documents(
        self,
        documents: List[Document],
        collection: str,
        **kwargs: Any
    ) -> List[str]:
        """Add documents to the store (async)."""
        return self.add_documents(documents, collection, **kwargs)
    
    def delete_documents(
        self,
        document_ids: List[str],
        collection: str,
        **kwargs: Any
    ) -> bool:
        """Delete documents from the store."""
        index = self._get_index()
        index.delete(ids=document_ids)
        return True
    
    def create_collection(
        self,
        name: str,
        dimension: int,
        **kwargs: Any
    ) -> bool:
        """Create a new collection."""
        # Pinecone doesn't have explicit collections, 
        # we use metadata filtering for collection-like behavior
        return True
    
    def list_collections(self) -> List[CollectionInfo]:
        """List all collections.""" 
        # Pinecone doesn't support listing collections directly
        # This would require querying with different collection filters
        return []
    
    def get_collection_info(self, name: str) -> Optional[CollectionInfo]:
        """Get collection information."""
        index = self._get_index()
        stats = index.describe_index_stats()
        
        return CollectionInfo(
            name=name,
            document_count=stats.total_vector_count,
            vector_count=stats.total_vector_count,
            created_at="",
            updated_at="",
            metadata={}
        )


def create_vector_store(store_type: str, **config: Any) -> VectorStore:
    """Factory function to create vector stores."""
    store_map = {
        "chroma": ChromaVectorStore,
        "pinecone": PineconeVectorStore,
    }
    
    if store_type not in store_map:
        raise ValueError(f"Unsupported vector store: {store_type}")
    
    return store_map[store_type](**config)