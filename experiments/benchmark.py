"""
Benchmark script for POS-Expo vs ERM vs Oracle.
Runs on 3 scenarios: 1D Linear (Toy), HD Gauss, and Adult (Real).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import time

from pos_expo.datasets import make_toy_selection_bias, make_toy_hd_gauss, load_adult_income
from pos_expo.features import build_default_feature_map
from pos_expo.estimators import PosExpoClassifier

def run_scenario(name, data_gen_fn, n_runs=3):
    results = []
    
    print(f"\n--- Running Scenario: {name} ---")
    
    for i in range(n_runs):
        seed = 42 + i
        
        # 1. Data Generation
        if name == "Adult":
            try:
                X_train, y_train, X_test, y_test, X_ref = data_gen_fn(biased=True, random_state=seed)
                iw_star = None # No oracle for real data
                if X_ref is None: X_ref = X_test
            except Exception as e:
                print(f"Skipping Adult: {e}")
                return
        else:
            # Synthetic
            X_train, y_train, X_test, y_test, iw_star = data_gen_fn(n_samples=2000, random_state=seed)
            X_ref = X_test # Use test as proxy for target population
            
        # 2. ERM
        t0 = time.time()
        erm = LogisticRegression(max_iter=1000)
        erm.fit(X_train, y_train)
        loss_erm = log_loss(y_test, erm.predict_proba(X_test))
        t_erm = time.time() - t0
        
        # 3. IW* (Oracle)
        loss_iw = np.nan
        if iw_star is not None:
            iw_model = LogisticRegression(max_iter=1000)
            iw_model.fit(X_train, y_train, sample_weight=iw_star)
            loss_iw = log_loss(y_test, iw_model.predict_proba(X_test))
            
        # 4. POS-Expo
        t0 = time.time()
        # Use a slightly stronger reg for real data/HD
        alpha_reg = 1.0 if name == "Adult" else 0.1
        
        clf = PosExpoClassifier(
            base_estimator=LogisticRegression(max_iter=1000),
            feature_map=build_default_feature_map(),
            alpha_reg=alpha_reg
        )
        clf.fit(X_train, y_train, X_ref=X_ref)
        loss_pos = log_loss(y_test, clf.predict_proba(X_test))
        t_pos = time.time() - t0
        
        results.append({
            "run": i,
            "loss_erm": loss_erm,
            "loss_iw": loss_iw,
            "loss_pos": loss_pos,
            "time_erm": t_erm,
            "time_pos": t_pos
        })
        
    # Aggregate
    df = pd.DataFrame(results)
    print(df.mean(numeric_only=True))
    return df

def main():
    # 1. Toy 1D
    run_scenario("1D_Piecewise", make_toy_selection_bias)
    
    # 2. HD Gauss
    run_scenario("HD_Gauss", lambda **k: make_toy_hd_gauss(n_features=20, **k))
    
    # 3. Adult
    run_scenario("Adult", load_adult_income)

if __name__ == "__main__":
    main()

