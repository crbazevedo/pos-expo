import numpy as np
import pytest
from pos_expo.core import StructuralFeatureMap
from pos_expo.tilt import PosExpoReweighter

class MockFeatureMap:
    """Simple feature map for property testing."""
    def fit(self, X, y=None, base_model=None):
        return self
    
    def transform(self, X, y=None, base_model=None):
        return X  # Just return X as features
        
    def fit_transform(self, X, y=None, base_model=None):
        return X

def test_alpha_zero_behavior():
    """Test that alpha=0 implies weights approx 1."""
    X = np.random.randn(100, 2)
    fm = MockFeatureMap()
    reweighter = PosExpoReweighter(feature_map=fm)
    
    # Manually set alpha to 0
    reweighter.alpha_ = np.zeros(2)
    reweighter.bias_ = 0.0
    
    # Compute weights without normalization first to check raw exp(0)
    # But compute_weights might normalize by default.
    # Let's check raw scores if possible, or check normalized result.
    # If alpha=0, w(z) = exp(0) = 1. Mean is 1. Normalized is 1.
    
    weights = reweighter.compute_weights(X, normalize=True)
    
    assert np.allclose(weights, 1.0)
    assert np.allclose(np.mean(weights), 1.0)

def test_monotonicity():
    """Test that if alpha >= 0, z_i <= z_j implies w_i <= w_j."""
    # 1D features: x in R
    # z1 = 1, z2 = 2. z1 <= z2.
    # If alpha > 0, then alpha*z1 < alpha*z2 => exp(...) < exp(...)
    
    X = np.array([[1.0], [2.0]])
    fm = MockFeatureMap()
    reweighter = PosExpoReweighter(feature_map=fm, allow_negative_alpha=False)
    
    # Set positive alpha
    reweighter.alpha_ = np.array([1.0])
    reweighter.bias_ = 0.0
    
    weights = reweighter.compute_weights(X, normalize=False)
    
    assert weights[0] < weights[1]
    
    # Check strict inequality for strictly larger input
    assert weights[1] > weights[0] + 1e-6

def test_normalization():
    """Test that weights satisfy mean(w) = 1 when requested."""
    X = np.random.randn(50, 2)
    fm = MockFeatureMap()
    reweighter = PosExpoReweighter(feature_map=fm)
    
    # Set arbitrary alpha
    reweighter.alpha_ = np.array([0.5, -0.2])
    reweighter.bias_ = 0.1
    
    weights = reweighter.compute_weights(X, normalize=True)
    
    assert np.isclose(np.mean(weights), 1.0)
    assert np.all(weights > 0)
