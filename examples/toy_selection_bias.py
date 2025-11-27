"""
Example script demonstrating POS-Expo on a toy selection bias problem.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from pos_expo.datasets import make_toy_selection_bias
from pos_expo.features import build_default_feature_map
from pos_expo.estimators import PosExpoClassifier
from pos_expo.diagnostics import weight_statistics, covariate_shift_summary

def main():
    # 1. Generate Data
    print("Generating data...")
    X_train, y_train, X_test, y_test, _ = make_toy_selection_bias(n_samples=2000, random_state=42)
    X_ref = X_test  # In this toy setup, we have access to target distribution samples
    
    print(f"Train size: {X_train.shape[0]}")
    print(f"Test size: {X_test.shape[0]}")
    
    # 2. ERM (Baseline)
    print("\nTraining ERM (Logistic Regression)...")
    erm = LogisticRegression()
    erm.fit(X_train, y_train)
    loss_erm = log_loss(y_test, erm.predict_proba(X_test))
    print(f"ERM Test Log Loss: {loss_erm:.4f}")
    
    # 3. POS-Expo
    print("\nTraining POS-Expo...")
    # Define features: Rarity, Difficulty (using base ERM model), Diversity
    # We pass the ERM model as base_model to features if needed (DifficultyFeature uses it)
    # Note: PosExpoClassifier will fit the feature map.
    # DifficultyFeature needs a base_model. We can pass the pre-trained ERM or let it handle it.
    # The default DifficultyFeature expects a base_model to compute loss. 
    # PosExpoClassifier currently doesn't automatically train a separate model for DifficultyFeature 
    # before reweighting.
    # Strategy: Pass a pre-trained simple model to DifficultyFeature?
    # Or just use Diversity/Rarity.
    
    # Let's use a composite map.
    feature_map = build_default_feature_map()
    
    # For difficulty, we might want to populate it. 
    # Currently PosExpoClassifier.fit fits the reweighter.
    # The reweighter calls feature_map.fit_transform(X, y, base_model=None).
    # So DifficultyFeature will see None and return 0s. 
    # To use Difficulty, we should ideally pass a base_model to fit/transform.
    # PosExpoClassifier doesn't expose passing base_model to fit yet.
    # This is a refinement for future. For now, Difficulty will be 0.
    
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(),
        feature_map=feature_map,
        alpha_reg=0.1
    )
    
    clf.fit(X_train, y_train, X_ref=X_ref)
    
    loss_posexpo = log_loss(y_test, clf.predict_proba(X_test))
    print(f"POS-Expo Test Log Loss: {loss_posexpo:.4f}")
    
    # 4. Diagnostics
    print("\nDiagnostics:")
    weights = clf.reweighter_.compute_weights(X_train)
    stats = weight_statistics(weights)
    print("Weight Stats:", stats)
    
    print(f"Learned Alpha: {clf.reweighter_.alpha_}")

if __name__ == "__main__":
    main()

