"""Embedding provider abstraction for RAG functionality."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from justllms.core.models import ModelInfo, ProviderConfig
from justllms.retrieval.models import Document


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, api_key: str, **kwargs: Any):
        self.api_key = api_key
        self.config = kwargs
    
    @abstractmethod
    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    @abstractmethod
    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        """Generate embedding for a single text (async).""" 
        pass
    
    @abstractmethod
    async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for multiple texts (async)."""
        pass
    
    def embed_documents(self, documents: List[Document], **kwargs: Any) -> List[Document]:
        """Embed documents and update their embedding field."""
        texts = [doc.content for doc in documents]
        embeddings = self.embed_texts(texts, **kwargs)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            
        return documents
    
    async def aembed_documents(self, documents: List[Document], **kwargs: Any) -> List[Document]:
        """Embed documents and update their embedding field (async)."""
        texts = [doc.content for doc in documents]
        embeddings = await self.aembed_texts(texts, **kwargs)
        
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
            
        return documents


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    MODELS = {
        "text-embedding-3-large": ModelInfo(
            name="text-embedding-3-large",
            provider="openai",
            max_tokens=8192,
            max_context_length=8192,
            supports_functions=False,
            supports_vision=False,
            supports_streaming=False,
            cost_per_1k_prompt_tokens=0.00013,
            cost_per_1k_completion_tokens=0.0,
            tags=["embedding", "large", "3072-dim"],
        ),
        "text-embedding-3-small": ModelInfo(
            name="text-embedding-3-small", 
            provider="openai",
            max_tokens=8192,
            max_context_length=8192,
            supports_functions=False,
            supports_vision=False,
            supports_streaming=False,
            cost_per_1k_prompt_tokens=0.00002,
            cost_per_1k_completion_tokens=0.0,
            tags=["embedding", "small", "1536-dim"],
        ),
        "text-embedding-ada-002": ModelInfo(
            name="text-embedding-ada-002",
            provider="openai",
            max_tokens=8192,
            max_context_length=8192,
            supports_functions=False,
            supports_vision=False,
            supports_streaming=False,
            cost_per_1k_prompt_tokens=0.0001,
            cost_per_1k_completion_tokens=0.0,
            tags=["embedding", "legacy", "1536-dim"],
        ),
    }
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-large", **kwargs: Any):
        super().__init__(api_key, **kwargs)
        self.model = model
        self._client = None
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise ImportError("OpenAI package not installed. Install with: pip install openai") from e
        return self._client
    
    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """Generate embedding for single text."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=[text],
            **kwargs
        )
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        client = self._get_client()
        response = client.embeddings.create(
            model=self.model,
            input=texts,
            **kwargs
        )
        return [item.embedding for item in response.data]
    
    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        """Generate embedding for single text (async)."""
        client = self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=[text],
            **kwargs
        )
        return response.data[0].embedding
    
    async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for multiple texts (async).""" 
        client = self._get_client()
        response = await client.embeddings.create(
            model=self.model,
            input=texts,
            **kwargs
        )
        return [item.embedding for item in response.data]


class GoogleEmbeddingProvider(EmbeddingProvider):
    """Google embedding provider."""
    
    MODELS = {
        "embedding-001": ModelInfo(
            name="embedding-001",
            provider="google",
            max_tokens=2048,
            max_context_length=2048, 
            supports_functions=False,
            supports_vision=False,
            supports_streaming=False,
            cost_per_1k_prompt_tokens=0.0001,  # Estimate
            cost_per_1k_completion_tokens=0.0,
            tags=["embedding", "768-dim"],
        ),
        "text-embedding-004": ModelInfo(
            name="text-embedding-004",
            provider="google", 
            max_tokens=2048,
            max_context_length=2048,
            supports_functions=False,
            supports_vision=False,
            supports_streaming=False,
            cost_per_1k_prompt_tokens=0.00001,
            cost_per_1k_completion_tokens=0.0,
            tags=["embedding", "768-dim", "latest"],
        ),
    }
    
    def __init__(self, api_key: str, model: str = "text-embedding-004", **kwargs: Any):
        super().__init__(api_key, **kwargs)
        self.model = model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def embed_text(self, text: str, **kwargs: Any) -> List[float]:
        """Generate embedding for single text."""
        return self.embed_texts([text], **kwargs)[0]
    
    def embed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        import httpx
        
        embeddings = []
        for text in texts:
            url = f"{self.base_url}/models/{self.model}:embedContent"
            headers = {"Content-Type": "application/json"}
            params = {"key": self.api_key}
            
            data = {
                "content": {"parts": [{"text": text}]},
                "taskType": kwargs.get("task_type", "RETRIEVAL_DOCUMENT"),
            }
            
            with httpx.Client() as client:
                response = client.post(url, headers=headers, params=params, json=data)
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"]["values"])
        
        return embeddings
    
    async def aembed_text(self, text: str, **kwargs: Any) -> List[float]:
        """Generate embedding for single text (async)."""
        results = await self.aembed_texts([text], **kwargs)
        return results[0]
    
    async def aembed_texts(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        """Generate embeddings for multiple texts (async)."""
        import httpx
        
        embeddings = []
        async with httpx.AsyncClient() as client:
            for text in texts:
                url = f"{self.base_url}/models/{self.model}:embedContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.api_key}
                
                data = {
                    "content": {"parts": [{"text": text}]},
                    "taskType": kwargs.get("task_type", "RETRIEVAL_DOCUMENT"),
                }
                
                response = await client.post(url, headers=headers, params=params, json=data)
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"]["values"])
        
        return embeddings


def create_embedding_provider(provider: str, **config: Any) -> EmbeddingProvider:
    """Factory function to create embedding providers."""
    provider_map = {
        "openai": OpenAIEmbeddingProvider,
        "google": GoogleEmbeddingProvider,
    }
    
    if "/" in provider:
        provider_name, model = provider.split("/", 1)
        config["model"] = model
    else:
        provider_name = provider
    
    if provider_name not in provider_map:
        raise ValueError(f"Unsupported embedding provider: {provider_name}")
    
    return provider_map[provider_name](**config)