"""
Example script demonstrating POS-Expo on the Adult Income dataset.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

from pos_expo.datasets import load_adult_income
from pos_expo.features import build_default_feature_map
from pos_expo.estimators import PosExpoClassifier
from pos_expo.diagnostics import weight_statistics, covariate_shift_summary

def main():
    print("Loading Adult Income Dataset...")
    # This will use the dummy fallback if download fails, which is fine for demo
    X_train, y_train, X_test, y_test, X_ref = load_adult_income(biased=True)
    
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    
    if X_ref is None:
        print("No reference data available (unbiased setting?). Using X_test as proxy for target.")
        X_ref = X_test

    # 1. ERM
    print("\nTraining ERM...")
    erm = LogisticRegression(max_iter=1000)
    erm.fit(X_train, y_train)
    
    acc_erm = accuracy_score(y_test, erm.predict(X_test))
    loss_erm = log_loss(y_test, erm.predict_proba(X_test))
    print(f"ERM Accuracy: {acc_erm:.4f}, Log Loss: {loss_erm:.4f}")
    
    # 2. POS-Expo
    print("\nTraining POS-Expo...")
    # Using simpler map for speed/robustness on real data
    feature_map = build_default_feature_map(rarity_k=20, diversity_clusters=20)
    
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(max_iter=1000),
        feature_map=feature_map,
        alpha_reg=1.0 # Stronger regularization for real data
    )
    
    clf.fit(X_train, y_train, X_ref=X_ref)
    
    acc_pos = accuracy_score(y_test, clf.predict(X_test))
    loss_pos = log_loss(y_test, clf.predict_proba(X_test))
    print(f"POS-Expo Accuracy: {acc_pos:.4f}, Log Loss: {loss_pos:.4f}")
    
    # 3. Diagnostics
    print("\nDiagnostics:")
    weights = clf.reweighter_.compute_weights(X_train)
    stats = weight_statistics(weights)
    print("Weight Stats:", stats)
    
    # Check shift correction
    shift_summary = covariate_shift_summary(X_train, X_ref, weights, feature_map)
    print(f"Original Shift (L2): {shift_summary['original_shift_l2']:.4f}")
    print(f"Corrected Shift (L2): {shift_summary['shift_correction_l2']:.4f}")

if __name__ == "__main__":
    main()


