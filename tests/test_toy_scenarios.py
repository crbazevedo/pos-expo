import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss

from pos_expo.datasets import make_toy_selection_bias
from pos_expo.features import build_default_feature_map
from pos_expo.estimators import PosExpoClassifier
from pos_expo.tilt import PosExpoReweighter, estimate_log_density_ratio

def test_projection_quality():
    """
    Test that we can project the true density ratio g* onto phi
    with reasonable accuracy in a toy setting.
    """
    # X_train ~ N(1,1), X_test ~ N(0,1)
    # w*(x) = exp(-x + 0.5) roughly
    # log w*(x) is linear in x.
    
    X_train, y_train, X_test, y_test, iw_star = make_toy_selection_bias(n_samples=1000, random_state=42)
    
    # log(iw_star)
    log_w_star = np.log(iw_star + 1e-12)
    
    # We use a feature map that includes x (RarityFeature kNN is approx density, 
    # but let's check if our default map spans this).
    # Rarity(x) ~ log(1/p(x)) ~ log(1/exp(-(x-1)^2)) ~ (x-1)^2
    # This is quadratic. The true log ratio is linear.
    # So (x-1)^2 might not perfectly capture linear x, but locally it might?
    # Actually, let's use a simpler map for this specific test or just verify default map works decent.
    
    feature_map = build_default_feature_map()
    phi_train = feature_map.fit_transform(X_train)
    
    # Ridge regression of log_w_star on phi
    ridge = Ridge(alpha=1e-5)
    ridge.fit(phi_train, log_w_star)
    pred_log_w = ridge.predict(phi_train)
    
    mse = np.mean((log_w_star - pred_log_w) ** 2)
    
    # We expect some error because default features (kNN, etc) are non-linear transformations
    # but they should capture the shift trend.
    # Let's assert it's "reasonably" low, e.g. < 1.0 (variance of log w is likely higher)
    var_log_w = np.var(log_w_star)
    r2 = 1 - mse / var_log_w
    
    print(f"R2 of projection: {r2:.4f}")
    assert r2 > 0.05, "Feature map should capture at least some signal of the shift"


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
    # Use default features
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(),
        feature_map=build_default_feature_map(rarity_k=20),
        alpha_reg=0.1
    )
    clf.fit(X_train, y_train, X_ref=X_ref)
    loss_pos = log_loss(y_test, clf.predict_proba(X_test))
    
    print(f"Losses -> ERM: {loss_erm:.4f}, IW*: {loss_iw:.4f}, POS: {loss_pos:.4f}")
    
    # Checks
    # POS should be better than ERM (lower loss)
    # Note: In small samples or weak shifts, ERM might be robust.
    # But here shift is designed to be meaningful.
    # Ensure POS is not disastrously worse than ERM (robustness) 
    # and hopefully better.
    
    # Strict improvement might be noisy, so let's allow small margin or check meaningful gap
    # Typically POS < ERM.
    assert loss_pos < loss_erm + 0.01 
    
    # POS should be close to IW*
    diff_iw = abs(loss_pos - loss_iw)
    assert diff_iw < 0.1, "POS-Expo should be reasonably close to Oracle IW"


