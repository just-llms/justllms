"""Qwen3 embedding service with local model support."""

import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class Qwen3EmbeddingService:
    """Qwen3 embedding service with caching and on-demand model download."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for Qwen3EmbeddingService. "
                "Install with: pip install transformers torch"
            )

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Setup cache
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.justllms/embeddings_cache")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._init_cache_db()

        # Lazy load model and tokenizer
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None

    def _init_cache_db(self) -> None:
        """Initialize SQLite cache database."""
        self.cache_db_path = self.cache_dir / "embeddings.db"
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT PRIMARY KEY,
                    model_name TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

    def _load_model(self) -> None:
        """Load model and tokenizer on first use."""
        if self.model is not None:
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            if self.model is not None:
                self.model.to(self.device)
                self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load {self.model_name}: {e}") from e

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self.model_name}:{text}".encode()).hexdigest()

    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        cache_key = self._get_cache_key(text)

        with sqlite3.connect(self.cache_db_path) as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash = ? AND model_name = ?",
                (cache_key, self.model_name),
            )
            result = cursor.fetchone()

        if result:
            return np.array(json.loads(result[0]), dtype=np.float32)
        return None

    def _cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Cache embedding to database."""
        cache_key = self._get_cache_key(text)
        embedding_json = json.dumps(embedding.tolist())

        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model_name, embedding) VALUES (?, ?, ?)",
                (cache_key, self.model_name, embedding_json),
            )

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using the model."""
        self._load_model()

        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Model or tokenizer not loaded")

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling on last hidden states
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embedding.cpu().numpy()

    def embed(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        # Check cache first
        cached = self._get_cached_embedding(text)
        if cached is not None:
            return cached

        # Generate new embedding
        embedding = self._generate_embedding(text)

        # Cache the result
        self._cache_embedding(text, embedding)

        return embedding

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed(text))
        return embeddings

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        with sqlite3.connect(self.cache_db_path) as conn:
            conn.execute("DELETE FROM embeddings WHERE model_name = ?", (self.model_name,))
