from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin

from ..core import Array, StructuralFeatureMap

class RarityFeature(BaseEstimator, TransformerMixin):
    """
    Computes rarity score based on kNN distance.
    rarity(x) = log(mean distance to k nearest neighbors).
    
    Higher score = rarer (lower density).
    """

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.nbrs_: Optional[NearestNeighbors] = None

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
    ) -> "RarityFeature":
        self.nbrs_ = NearestNeighbors(n_neighbors=self.k).fit(X)
        return self

    def transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[object] = None,
    ) -> Array:
        if self.nbrs_ is None:
            raise RuntimeError("RarityFeature not fitted.")
            
        distances, _ = self.nbrs_.kneighbors(X)
        # distances shape: (n_samples, k)
        
        # Use mean distance as sparsity measure
        mean_dist = distances.mean(axis=1)
        
        # Avoid log(0)
        eps = 1e-12
        rarity = np.log(np.maximum(mean_dist, eps))
        
        return rarity.reshape(-1, 1)

