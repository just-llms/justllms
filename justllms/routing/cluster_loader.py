"""Cluster artifact loader for intelligent routing."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class ClusterArtifactLoader:
    """Loads and manages cluster artifacts for routing."""

    def __init__(self, artifacts_path: Optional[str] = None):
        if artifacts_path is None:
            # Default to bundled artifacts
            package_root = Path(__file__).parent.parent
            self.artifacts_path = package_root / "data" / "cluster_artifacts"
        else:
            self.artifacts_path = Path(artifacts_path)

        if not self.artifacts_path.exists():
            raise FileNotFoundError(f"Cluster artifacts not found at: {self.artifacts_path}")

        self._cluster_centers: Optional[np.ndarray] = None
        self._cluster_rankings: Optional[Dict[str, Any]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._normalizer: Optional[Any] = None

    @property
    def cluster_centers(self) -> np.ndarray:
        """Get cluster centers, loading if necessary."""
        if self._cluster_centers is None:
            centers_path = self.artifacts_path / "cluster_centers.npy"
            if not centers_path.exists():
                raise FileNotFoundError(f"Cluster centers not found: {centers_path}")
            self._cluster_centers = np.load(centers_path)
        return self._cluster_centers

    @property
    def cluster_rankings(self) -> Dict[str, Any]:
        """Get cluster rankings, loading if necessary."""
        if self._cluster_rankings is None:
            rankings_path = self.artifacts_path / "cluster_rankings.json"
            if not rankings_path.exists():
                raise FileNotFoundError(f"Cluster rankings not found: {rankings_path}")

            with open(rankings_path) as f:
                self._cluster_rankings = json.load(f)
        return self._cluster_rankings

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata, loading if necessary."""
        if self._metadata is None:
            metadata_path = self.artifacts_path / "metadata.json"
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")

            with open(metadata_path) as f:
                self._metadata = json.load(f)
        return self._metadata

    @property
    def normalizer(self) -> Optional[Any]:
        """Get normalizer, loading if necessary."""
        if not JOBLIB_AVAILABLE:
            return None

        if self._normalizer is None:
            normalizer_path = self.artifacts_path / "normalizer.joblib"
            if normalizer_path.exists():
                self._normalizer = joblib.load(normalizer_path)
        return self._normalizer

    def get_cluster_ranking(self, cluster_id: int) -> List[str]:
        """Get model ranking for a specific cluster."""
        cluster_str = str(cluster_id)
        rankings = self.cluster_rankings

        if cluster_str not in rankings:
            raise ValueError(f"Cluster {cluster_id} not found in rankings")

        ranking = rankings[cluster_str]["ranking"]
        return ranking if isinstance(ranking, list) else []

    def get_cluster_scores(self, cluster_id: int) -> Dict[str, float]:
        """Get model scores for a specific cluster."""
        cluster_str = str(cluster_id)
        rankings = self.cluster_rankings

        if cluster_str not in rankings:
            raise ValueError(f"Cluster {cluster_id} not found in rankings")

        scores = rankings[cluster_str]["scores"]
        return scores if isinstance(scores, dict) else {}

    def get_available_models(self) -> List[str]:
        """Get list of all available models from metadata."""
        models = self.metadata.get("available_models", [])
        return models if isinstance(models, list) else []

    def get_n_clusters(self) -> int:
        """Get number of clusters."""
        n_clusters = self.metadata.get("n_clusters", len(self.cluster_centers))
        return n_clusters if isinstance(n_clusters, int) else len(self.cluster_centers)

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding using the saved normalizer or fallback to L2."""
        if self.normalizer is not None:
            return self.normalizer.transform(embedding.reshape(1, -1))[0]
        else:
            # Fallback to L2 normalization
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding

    def find_closest_clusters(
        self, query_embedding: np.ndarray, top_k: int = 1
    ) -> List[Tuple[int, float]]:
        """Find closest clusters to query embedding."""
        normalized_query = self.normalize_embedding(query_embedding)

        # Compute cosine similarities
        similarities = np.dot(self.cluster_centers, normalized_query)

        # Get top-k closest clusters
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def validate_artifacts(self) -> bool:
        """Validate that all required artifacts are present and valid."""
        try:
            # Check all files exist
            required_files = ["cluster_centers.npy", "cluster_rankings.json", "metadata.json"]
            for filename in required_files:
                if not (self.artifacts_path / filename).exists():
                    return False

            # Validate cluster centers shape
            centers = self.cluster_centers
            if len(centers.shape) != 2:
                return False

            # Validate rankings structure
            rankings = self.cluster_rankings
            n_clusters = self.get_n_clusters()

            for i in range(n_clusters):
                if str(i) not in rankings:
                    return False

                cluster_data = rankings[str(i)]
                if "ranking" not in cluster_data or "scores" not in cluster_data:
                    return False

            return True
        except Exception:
            return False

    def get_model_provider_mapping(self) -> Dict[str, Tuple[str, str]]:
        """Get mapping from model names to (provider, model) tuples."""
        mapping = {}
        for model in self.get_available_models():
            if "/" in model:
                provider, model_name = model.split("/", 1)
                mapping[model] = (provider, model_name)
            else:
                # Fallback: assume it's just a model name
                mapping[model] = ("unknown", model)
        return mapping
