import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from pos_expo.datasets import load_adult_income
from pos_expo.estimators import PosExpoClassifier
from pos_expo.features import build_default_feature_map

def test_adult_smoke_pipeline():
    """
    Smoke test on a subsample of Adult dataset.
    Ensures pipeline runs and produces sane results (not worse than ERM).
    """
    # 1. Load subsample
    # Biased training data
    try:
        X_train, y_train, X_test, y_test, X_ref = load_adult_income(
            biased=True, 
            test_size=0.3, 
            random_state=42
        )
    except Exception:
        pytest.skip("Adult dataset could not be loaded/downloaded.")
        
    # Subsample for speed
    n_sub = 2000
    if len(X_train) > n_sub:
        X_train = X_train[:n_sub]
        y_train = y_train[:n_sub]
    if len(X_test) > n_sub:
        X_test = X_test[:n_sub]
        y_test = y_test[:n_sub]
    if X_ref is not None and len(X_ref) > n_sub:
        X_ref = X_ref[:n_sub]
        
    if X_ref is None:
        X_ref = X_test
        
    # 2. Train ERM
    erm = LogisticRegression(max_iter=500)
    erm.fit(X_train, y_train)
    loss_erm = log_loss(y_test, erm.predict_proba(X_test))
    
    # 3. Train POS-Expo
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(max_iter=500),
        feature_map=build_default_feature_map(),
        alpha_reg=1.0
    )
    clf.fit(X_train, y_train, X_ref=X_ref)
    loss_pos = log_loss(y_test, clf.predict_proba(X_test))
    
    print(f"Adult Smoke - ERM: {loss_erm:.4f}, POS: {loss_pos:.4f}")
    
    # 4. Assert Robustness
    # Should not be significantly worse than ERM
    # (Allowing slight degradation due to noise/hyperparams, e.g. 0.05)
    assert loss_pos < loss_erm + 0.05
    
    # Check weights are not degenerate
    weights = clf.reweighter_.compute_weights(X_train)
    assert np.all(weights > 0)
    assert np.max(weights) < 1000 # Sanity check for explosion
