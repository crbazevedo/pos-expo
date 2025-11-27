from typing import Optional, List
import numpy as np

from ..core import StructuralFeatureMap, CompositeFeatureMap
from .rarity import RarityFeature
from .difficulty import DifficultyFeature
from .diversity import DiversityFeature

__all__ = [
    "RarityFeature",
    "DifficultyFeature",
    "DiversityFeature",
    "build_default_feature_map",
]

def build_default_feature_map(
    rarity_k: int = 5,
    diversity_clusters: int = 10,
) -> StructuralFeatureMap:
    """
    Builds the default CompositeFeatureMap with:
    1. RarityFeature (kNN)
    2. DifficultyFeature (Loss/Entropy)
    3. DiversityFeature (KMeans)
    """
    return CompositeFeatureMap([
        RarityFeature(k=rarity_k),
        DifficultyFeature(),
        DiversityFeature(n_clusters=diversity_clusters),
    ])
