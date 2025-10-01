from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from justllms.core.base import BaseProvider
from justllms.core.models import Message


class RoutingStrategy(ABC):
    """Abstract base class for routing strategies."""

    @abstractmethod
    def select(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select a provider and model based on the strategy.

        Returns:
            Tuple of (provider_name, model_name)
        """
        pass


class ClusterBasedStrategy(RoutingStrategy):
    """Intelligent cluster-based routing using pre-trained embeddings.

    This strategy uses lazy loading for heavy dependencies (transformers, torch, numpy)
    to ensure the library works with base installation. Dependencies are only loaded
    when cluster routing is actually used.
    """

    def __init__(
        self,
        artifacts_path: Optional[str] = None,
        top_k_clusters: int = 1,
        similarity_threshold: float = 0.0,
        enable_logging: bool = False,
    ):
        # Store config but don't load heavy deps yet
        self.artifacts_path = artifacts_path
        self.top_k_clusters = top_k_clusters
        self.similarity_threshold = similarity_threshold
        self.enable_logging = enable_logging

        # Lazy-loaded components
        self._embedding_service: Optional[Any] = None
        self._cluster_loader: Optional[Any] = None
        self._provider_mapping: Optional[Dict[str, Tuple[str, str]]] = None
        self._init_attempted = False
        self._init_failed = False
        self._init_error: Optional[str] = None

    def _lazy_init(self) -> bool:
        """Lazy initialization of cluster components.

        Returns:
            bool: True if initialization succeeded, False otherwise.
        """
        if self._init_attempted:
            return not self._init_failed

        self._init_attempted = True

        try:
            # Import heavy dependencies only when needed
            from ..embeddings import Qwen3EmbeddingService
            from .cluster_loader import ClusterArtifactLoader

            # Load cluster artifacts
            try:
                self._cluster_loader = ClusterArtifactLoader(self.artifacts_path)
                if not self._cluster_loader.validate_artifacts():
                    raise ValueError("Invalid cluster artifacts")
            except Exception as e:
                self._init_failed = True
                self._init_error = f"Failed to load cluster artifacts: {e}"
                if self.enable_logging:
                    print(f"ClusterBasedStrategy: {self._init_error}")
                return False

            # Initialize embedding service
            try:
                self._embedding_service = Qwen3EmbeddingService()
            except Exception as e:
                self._init_failed = True
                self._init_error = f"Failed to initialize embedding service: {e}"
                if self.enable_logging:
                    print(f"ClusterBasedStrategy: {self._init_error}")
                return False

            if self.enable_logging:
                print("ClusterBasedStrategy: Successfully initialized cluster routing")

            return True

        except ImportError as e:
            self._init_failed = True
            self._init_error = (
                f"Cluster routing requires additional dependencies. "
                f"Install with: pip install justllms[cluster]\n"
                f"Missing: {e.name if hasattr(e, 'name') else str(e)}"
            )
            if self.enable_logging:
                print(f"ClusterBasedStrategy: {self._init_error}")
            return False
        except Exception as e:
            self._init_failed = True
            self._init_error = f"Unexpected error during initialization: {e}"
            if self.enable_logging:
                print(f"ClusterBasedStrategy: {self._init_error}")
            return False

    def _get_provider_mapping(self) -> Dict[str, Tuple[str, str]]:
        """Get cached provider mapping."""
        if self._provider_mapping is None and self._cluster_loader is not None:
            self._provider_mapping = self._cluster_loader.get_model_provider_mapping()
        return self._provider_mapping or {}

    def _extract_query_text(self, messages: List[Message]) -> str:
        """Extract text content from messages for embedding."""
        texts = []
        for msg in messages:
            if isinstance(msg.content, str):
                texts.append(msg.content)
            elif isinstance(msg.content, list):
                # Handle multimodal messages - extract text parts
                for item in msg.content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        texts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        texts.append(item)

        return " ".join(texts).strip()

    def _fallback_to_configured_or_first_available(
        self,
        providers: Dict[str, BaseProvider],
        fallback_provider: Optional[str] = None,
        fallback_model: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Fallback to configured fallback or first available model.

        Args:
            providers: Available providers
            fallback_provider: Configured fallback provider from Router
            fallback_model: Configured fallback model from Router

        Returns:
            Tuple of (provider_name, model_name)
        """
        # First, try configured fallback if provided
        if fallback_provider and fallback_model and fallback_provider in providers:
            provider = providers[fallback_provider]
            available_models = provider.get_available_models()
            if fallback_model in available_models:
                if self.enable_logging:
                    print(
                        f"ClusterBasedStrategy: Using configured fallback {fallback_provider}/{fallback_model}"
                    )
                return fallback_provider, fallback_model

        # Fall back to first available model
        for provider_name, provider in providers.items():
            models = provider.get_available_models()
            if models:
                model_name = list(models.keys())[0]
                if self.enable_logging:
                    print(
                        f"ClusterBasedStrategy: Fallback to first available {provider_name}/{model_name}"
                    )
                return provider_name, model_name

        raise ValueError("No models available in any provider")

    def select(
        self,
        messages: List[Message],
        providers: Dict[str, BaseProvider],
        constraints: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Select provider and model using cluster-based routing.

        Expects fallback_provider and fallback_model to be passed via kwargs
        from the Router.

        Args:
            messages: List of messages to route
            providers: Available providers
            constraints: Optional routing constraints
            **kwargs: Additional arguments including fallback_provider and fallback_model

        Returns:
            Tuple of (provider_name, model_name)
        """
        # Extract fallback config from kwargs (passed by Router)
        fallback_provider = kwargs.get("fallback_provider")
        fallback_model = kwargs.get("fallback_model")

        # Try lazy initialization
        if not self._lazy_init():
            # Cluster routing unavailable, use fallback
            if self.enable_logging:
                print(
                    f"ClusterBasedStrategy: Cluster routing unavailable. "
                    f"Reason: {self._init_error}"
                )
            return self._fallback_to_configured_or_first_available(
                providers, fallback_provider, fallback_model
            )

        try:
            # Extract query text
            query_text = self._extract_query_text(messages)
            if not query_text:
                if self.enable_logging:
                    print("ClusterBasedStrategy: No text content found, falling back")
                return self._fallback_to_configured_or_first_available(
                    providers, fallback_provider, fallback_model
                )

            # Generate embedding
            if self._embedding_service is None:
                raise RuntimeError("Embedding service not initialized")

            query_embedding = self._embedding_service.embed(query_text)

            # Find closest clusters
            if self._cluster_loader is None:
                raise RuntimeError("Cluster loader not initialized")

            closest_clusters = self._cluster_loader.find_closest_clusters(
                query_embedding, top_k=self.top_k_clusters
            )

            if not closest_clusters:
                if self.enable_logging:
                    print("ClusterBasedStrategy: No clusters found, falling back")
                return self._fallback_to_configured_or_first_available(
                    providers, fallback_provider, fallback_model
                )

            # Check similarity threshold
            best_cluster_id, best_similarity = closest_clusters[0]
            if best_similarity < self.similarity_threshold:
                if self.enable_logging:
                    print(
                        f"ClusterBasedStrategy: Similarity {best_similarity:.3f} below threshold {self.similarity_threshold}, falling back"
                    )
                return self._fallback_to_configured_or_first_available(
                    providers, fallback_provider, fallback_model
                )

            # Get provider mapping
            provider_mapping = self._get_provider_mapping()

            # Try each cluster in order of similarity
            for cluster_id, similarity in closest_clusters:
                try:
                    # Get model ranking for this cluster
                    ranking = self._cluster_loader.get_cluster_ranking(cluster_id)

                    # Try models in order of performance
                    for model_name in ranking:
                        if model_name in provider_mapping:
                            provider_name, actual_model = provider_mapping[model_name]

                            # Check if provider is available
                            if provider_name in providers:
                                provider = providers[provider_name]
                                available_models = provider.get_available_models()

                                # Check if model is available
                                if actual_model in available_models:
                                    if self.enable_logging:
                                        print(
                                            f"ClusterBasedStrategy: Routed to cluster {cluster_id} (sim={similarity:.3f}), selected {provider_name}/{actual_model}"
                                        )

                                    # Store routing metadata for analysis
                                    kwargs.setdefault("_routing_metadata", {}).update(
                                        {
                                            "strategy": "cluster",
                                            "cluster_id": cluster_id,
                                            "similarity_score": similarity,
                                            "query_text": (
                                                query_text[:100] + "..."
                                                if len(query_text) > 100
                                                else query_text
                                            ),
                                        }
                                    )

                                    return provider_name, actual_model

                except Exception as e:
                    if self.enable_logging:
                        print(f"ClusterBasedStrategy: Error processing cluster {cluster_id}: {e}")
                    continue

            # If we get here, no suitable models found in any cluster
            if self.enable_logging:
                print("ClusterBasedStrategy: No suitable models found in clusters, falling back")
            return self._fallback_to_configured_or_first_available(
                providers, fallback_provider, fallback_model
            )

        except Exception as e:
            if self.enable_logging:
                print(f"ClusterBasedStrategy: Error in cluster routing: {e}, falling back")
            return self._fallback_to_configured_or_first_available(
                providers, fallback_provider, fallback_model
            )
