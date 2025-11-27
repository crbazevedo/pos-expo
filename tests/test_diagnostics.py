import numpy as np
from pos_expo.diagnostics import weight_statistics, structural_moments

def test_weight_statistics():
    weights = np.array([1, 1, 1, 1])
    stats = weight_statistics(weights)
    
    assert stats["mean"] == 1.0
    assert stats["min"] == 1.0
    assert stats["ess"] == 4.0 # For uniform weights, ESS = N

    weights_skewed = np.array([1, 0, 0, 0])
    stats_skewed = weight_statistics(weights_skewed)
    assert stats_skewed["ess"] == 1.0

def test_structural_moments():
    weights = np.array([1, 1])
    phi = np.array([[1, 2], [3, 4]])
    
    moments = structural_moments(weights, phi)
    
    # Mean of [1, 3] is 2, Mean of [2, 4] is 3
    assert np.allclose(moments["mean"], [2, 3])
    
    # Std: 1, 1
    assert np.allclose(moments["std"], [1, 1])


