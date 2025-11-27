import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from pos_expo.features import RarityFeature, DifficultyFeature, DiversityFeature, build_default_feature_map

def test_rarity_feature():
    # 1D data: dense around 0, sparse around 10
    X = np.concatenate([
        np.random.normal(0, 0.1, 100),
        np.random.normal(10, 0.1, 10)
    ]).reshape(-1, 1)
    
    rarity = RarityFeature(k=5)
    rarity.fit(X)
    scores = rarity.transform(X)
    
    # Points around 10 should have higher rarity (larger distance to neighbors)
    mean_dense = scores[:100].mean()
    mean_sparse = scores[100:].mean()
    
    assert mean_sparse > mean_dense

def test_difficulty_feature():
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    
    model = LogisticRegression()
    model.fit(X, y)
    
    diff = DifficultyFeature()
    
    # 1. Supervised difficulty
    scores = diff.transform(X, y, base_model=model)
    assert scores.shape == (4, 1)
    assert np.all(scores >= 0)
    
    # 2. Unsupervised difficulty (Entropy)
    scores_unsup = diff.transform(X, base_model=model)
    assert scores_unsup.shape == (4, 1)
    assert np.all(scores_unsup >= 0)

def test_diversity_feature():
    # Two clusters: 0 and 10
    X = np.array([[0], [0.1], [10], [10.1]])
    
    div = DiversityFeature(n_clusters=2, random_state=42)
    div.fit(X)
    scores = div.transform(X)
    
    # Points close to centers should have low scores
    assert np.all(scores < 1.0)
    
    # A point far away
    X_far = np.array([[5]])
    score_far = div.transform(X_far)
    assert score_far[0, 0] > scores.max()

def test_default_feature_map():
    X = np.random.randn(20, 2)
    fm = build_default_feature_map()
    fm.fit(X)
    out = fm.transform(X)
    
    # Rarity(1) + Difficulty(1) + Diversity(1) = 3 cols
    assert out.shape == (20, 3)

