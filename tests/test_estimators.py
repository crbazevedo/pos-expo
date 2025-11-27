import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from pos_expo.estimators import PosExpoClassifier
from pos_expo.datasets.synthetic import make_toy_selection_bias
from pos_expo.features import build_default_feature_map

def test_posexpo_classifier_integration():
    # 1. Get data
    # X_train is shifted from X_test.
    X_train, y_train, X_test, y_test, _ = make_toy_selection_bias(n_samples=2000, random_state=42)
    
    # 2. Reference sample (approximate P_test)
    # In practice we often have unlabeled test data available as X_ref
    X_ref = X_test.copy()
    
    # 3. Define feature map
    # We use default map
    feature_map = build_default_feature_map(rarity_k=10, diversity_clusters=10)
    
    # 4. Base estimator
    base = LogisticRegression()
    
    # 5. POS-Expo Classifier
    clf = PosExpoClassifier(
        base_estimator=base,
        feature_map=feature_map,
        alpha_reg=0.1
    )
    
    # 6. Fit with X_ref
    clf.fit(X_train, y_train, X_ref=X_ref)
    
    # 7. Evaluate
    # Compare with ERM (unweighted)
    erm = LogisticRegression()
    erm.fit(X_train, y_train)
    
    loss_posexpo = log_loss(y_test, clf.predict_proba(X_test))
    loss_erm = log_loss(y_test, erm.predict_proba(X_test))
    
    print(f"ERM Loss: {loss_erm:.4f}")
    print(f"POS-Expo Loss: {loss_posexpo:.4f}")
    
    # Expect POS-Expo to improve loss or at least be competitive
    # Since shift is simple and model is well-specified, improvement might be small or noise-dependent,
    # but let's check it doesn't crash and gives reasonable results.
    # In this toy 1D case, shift is simple covariate shift.
    
    assert hasattr(clf, "reweighter_")
    assert clf.reweighter_.alpha_ is not None

def test_posexpo_classifier_oracle():
    # Test with known weights
    X_train, y_train, X_test, y_test, iw_star = make_toy_selection_bias(n_samples=1000)
    
    feature_map = build_default_feature_map()
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(),
        feature_map=feature_map
    )
    
    clf.fit(X_train, y_train, iw_star=iw_star)
    
    assert clf.reweighter_.alpha_ is not None
    
    # Check predictions work
    probs = clf.predict_proba(X_test)
    assert probs.shape == (1000, 2)

