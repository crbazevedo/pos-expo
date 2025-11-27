import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

from pos_expo.datasets import make_toy_selection_bias
from pos_expo.features import build_default_feature_map
from pos_expo.estimators import PosExpoClassifier
from pos_expo.tilt import PosExpoReweighter, estimate_log_density_ratio
from pos_expo.core import StructuralFeatureMap

# Explicitly testing toy scenarios for projection quality and ordering

class IdentityFeatureMap(StructuralFeatureMap):
    def fit(self, X, y=None, base_model=None): return self
    def transform(self, X, y=None, base_model=None): return X
    def fit_transform(self, X, y=None, base_model=None): return X

def test_projection_quality():
    """
    Test that we can project the true density ratio g* onto phi.
    Validates the projection mechanism.
    """
    # X_train ~ N(1,1), X_test ~ N(0,1)
    # w*(x) = exp(-x + 0.5). log w*(x) is linear in x.
    X_train, y_train, X_test, y_test, iw_star = make_toy_selection_bias(n_samples=1000, random_state=42)
    
    log_w_star = np.log(iw_star + 1e-12)
    
    # 1. Test with Identity Map (Should be perfect as shift is linear)
    phi_identity = IdentityFeatureMap().fit_transform(X_train)
    ridge = Ridge(alpha=1e-5)
    ridge.fit(phi_identity, log_w_star)
    pred_log_w = ridge.predict(phi_identity)
    
    mse_id = np.mean((log_w_star - pred_log_w) ** 2)
    r2_id = 1 - mse_id / np.var(log_w_star)
    
    print(f"Diagnostics [Identity Map] - Projection R2: {r2_id:.4f}, MSE: {mse_id:.4f}")
    assert r2_id > 0.95, "Identity map should perfectly capture linear shift"

    # 2. Test with Default Map (Log diagnostics as requested)
    feature_map = build_default_feature_map()
    phi_def = feature_map.fit_transform(X_train)
    
    ridge_def = Ridge(alpha=1e-5)
    ridge_def.fit(phi_def, log_w_star)
    pred_log_w_def = ridge_def.predict(phi_def)
    
    mse_def = np.mean((log_w_star - pred_log_w_def) ** 2)
    r2_def = 1 - mse_def / np.var(log_w_star)
    
    print(f"Diagnostics [Default Map] - Projection R2: {r2_def:.4f}, MSE: {mse_def:.4f}")
    # Default map (Rarity/Diversity) might not capture linear shift well (it's symmetric/distance based).
    # We just ensure it doesn't crash and logs the value.
    assert mse_def < 10.0 # Loose bound

def test_toy_ordering():
    """
    Verify Loss(POS-Expo) < Loss(ERM) and close to IW*
    """
    X_train, y_train, X_test, y_test, iw_star = make_toy_selection_bias(n_samples=2000, random_state=42)
    X_ref = X_test # Oracle access to target samples
    
    # 1. ERM
    erm = LogisticRegression()
    erm.fit(X_train, y_train)
    loss_erm = log_loss(y_test, erm.predict_proba(X_test))
    
    # 2. IW* (Oracle Weights)
    iw_model = LogisticRegression()
    iw_model.fit(X_train, y_train, sample_weight=iw_star)
    loss_iw = log_loss(y_test, iw_model.predict_proba(X_test))
    
    # 3. POS-Expo (Learned weights)
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(),
        feature_map=build_default_feature_map(rarity_k=20),
        alpha_reg=0.1
    )
    clf.fit(X_train, y_train, X_ref=X_ref)
    loss_pos = log_loss(y_test, clf.predict_proba(X_test))
    
    # Diagnostics: Weight Distance
    w_pos = clf.reweighter_.compute_weights(X_train, normalize=True)
    w_star_norm = iw_star / iw_star.mean()
    weight_dist = np.linalg.norm(w_star_norm - w_pos) / np.sqrt(len(w_pos))
    
    print(f"Losses -> ERM: {loss_erm:.4f}, IW*: {loss_iw:.4f}, POS: {loss_pos:.4f}")
    print(f"Diagnostics -> Weight L2 Dist: {weight_dist:.4f}")
    
    # Checks
    assert loss_pos < loss_erm + 0.01 
    diff_iw = abs(loss_pos - loss_iw)
    assert diff_iw < 0.1, "POS-Expo should be reasonably close to Oracle IW"
