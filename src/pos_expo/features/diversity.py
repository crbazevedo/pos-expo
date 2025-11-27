from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin

from ..core import Array, StructuralFeatureMap

class DiversityFeature(BaseEstimator, TransformerMixin):
    """
    Computes diversity/coverage score.
    
    Approach: Cluster data (k-means), then feature is distance to nearest cluster center.
    High distance = unrepresented region (diverse relative to centers).
    """

    def __init__(self, n_clusters: int = 10, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_: Optional[KMeans] = None

    def fit_transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> Array:
        return self.fit(X, y, base_model).transform(X, y, base_model)

    def fit(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> "DiversityFeature":
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init="auto"
        ).fit(X)
        return self

    def transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> Array:
        if self.kmeans_ is None:
            raise RuntimeError("DiversityFeature not fitted.")
            
        # transform returns distance to each cluster center
        dists = self.kmeans_.transform(X)
        
        # We take distance to NEAREST center
        min_dist = dists.min(axis=1)
        
        return min_dist.reshape(-1, 1)

