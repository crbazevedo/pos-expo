import numpy as np
import pytest
from pos_expo.tilt import PosExpoReweighter
from pos_expo.datasets.synthetic import make_toy_selection_bias
from pos_expo.core import StructuralFeatureMap

class IdentityFeatureMap:
    def fit(self, X, y=None, base_model=None):
        return self
    def transform(self, X, y=None, base_model=None):
        return X
    def fit_transform(self, X, y=None, base_model=None):
        return X

def test_posexpo_reweighter_oracle():
    # 1. Generate data where we know w* depends on X
    # In make_toy_selection_bias:
    # X_train ~ N(1, 1)
    # X_test ~ N(0, 1)
    # w*(x) propto exp(-x)
    
    X_train, y_train, X_test, y_test, iw_star = make_toy_selection_bias(n_samples=500)
    
    # 2. Use IdentityFeatureMap because log(w*) is linear in x here!
    # log(w*(x)) = -x + 0.5
    # So if phi(x) = x, we should learn alpha approx -1.
    
    feature_map = IdentityFeatureMap()
    reweighter = PosExpoReweighter(feature_map=feature_map, alpha_reg=1e-5)
    
    reweighter.fit(X_train, iw_star=iw_star)
    
    # Check learned alpha
    # We expect alpha_ to be close to -1
    print(f"Learned alpha: {reweighter.alpha_}")
    assert np.isclose(reweighter.alpha_[0], -1.0, atol=0.1)
    
    # Check weights computation
    weights = reweighter.compute_weights(X_train, normalize=True)
    
    # Correlation between learned weights and true weights should be high
    corr = np.corrcoef(weights, iw_star)[0, 1]
    print(f"Correlation: {corr}")
    assert corr > 0.99
    
    # Check normalization
    assert np.isclose(np.mean(weights), 1.0)

def test_posexpo_reweighter_no_input_raises():
    X_train = np.random.randn(10, 2)
    reweighter = PosExpoReweighter(feature_map=IdentityFeatureMap())
    
    with pytest.raises(ValueError, match="Either iw_star or X_ref must be provided"):
        reweighter.fit(X_train, iw_star=None, X_ref=None)

