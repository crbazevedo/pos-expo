import numpy as np
from pos_expo.core import StructuralFeatureMap, CompositeFeatureMap

class MockFeatureMap:
    def __init__(self, mult=1.0):
        self.mult = mult
        
    def fit(self, X, y=None, base_model=None):
        return self
        
    def transform(self, X, y=None, base_model=None):
        return X * self.mult
        
    def fit_transform(self, X, y=None, base_model=None):
        return self.transform(X, y, base_model)

def test_composite_feature_map():
    X = np.ones((5, 1))
    
    fm1 = MockFeatureMap(mult=1.0)
    fm2 = MockFeatureMap(mult=2.0)
    
    comp = CompositeFeatureMap([fm1, fm2])
    
    out = comp.fit_transform(X)
    
    assert out.shape == (5, 2)
    assert np.allclose(out[:, 0], 1.0)
    assert np.allclose(out[:, 1], 2.0)

def test_composite_empty():
    X = np.ones((5, 1))
    comp = CompositeFeatureMap([])
    out = comp.fit_transform(X)
    assert out.shape == (5, 0)

