"""
Modified benchmark script to save results to CSVs for plotting.
Runs on 3 scenarios: 1D Linear (Toy), HD Gauss, and Adult (Real).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import time
import os

from pos_expo.datasets import make_toy_selection_bias, make_toy_hd_gauss, load_adult_income
from pos_expo.features import build_default_feature_map
from pos_expo.estimators import PosExpoClassifier

def run_scenario(name, data_gen_fn, n_runs=5):
    results = []
    
    print(f"\n--- Running Scenario: {name} ---")
    
    for i in range(n_runs):
        seed = 42 + i
        
        # 1. Data Generation
        if name == "Adult":
            try:
                X_train, y_train, X_test, y_test, X_ref = data_gen_fn(biased=True, random_state=seed)
                iw_star = None 
                if X_ref is None: X_ref = X_test
            except Exception as e:
                print(f"Skipping Adult: {e}")
                return None
        else:
            X_train, y_train, X_test, y_test, iw_star = data_gen_fn(n_samples=2000, random_state=seed)
            X_ref = X_test 
            
        # 2. ERM
        erm = LogisticRegression(max_iter=1000)
        erm.fit(X_train, y_train)
        loss_erm = log_loss(y_test, erm.predict_proba(X_test))
        
        # 3. IW* (Oracle)
        loss_iw = np.nan
        if iw_star is not None:
            iw_model = LogisticRegression(max_iter=1000)
            iw_model.fit(X_train, y_train, sample_weight=iw_star)
            loss_iw = log_loss(y_test, iw_model.predict_proba(X_test))
            
        # 4. POS-Expo
        alpha_reg = 1.0 if name == "Adult" else 0.1
        
        clf = PosExpoClassifier(
            base_estimator=LogisticRegression(max_iter=1000),
            feature_map=build_default_feature_map(),
            alpha_reg=alpha_reg
        )
        clf.fit(X_train, y_train, X_ref=X_ref)
        loss_pos = log_loss(y_test, clf.predict_proba(X_test))
        
        # Diagnostics
        weight_dist_l2 = np.nan
        if iw_star is not None:
            w_pos = clf.reweighter_.compute_weights(X_train, normalize=True)
            w_star_norm = iw_star / iw_star.mean()
            weight_dist_l2 = np.linalg.norm(w_star_norm - w_pos) / np.sqrt(len(w_pos))
        
        results.append({
            "scenario": name,
            "run": i,
            "loss_erm": loss_erm,
            "loss_iw": loss_iw,
            "loss_pos": loss_pos,
            "weight_dist_l2": weight_dist_l2,
        })
        
    df = pd.DataFrame(results)
    return df

def main():
    dfs = []
    
    # 1. Toy 1D
    df_1d = run_scenario("1D_Piecewise", make_toy_selection_bias)
    if df_1d is not None: dfs.append(df_1d)
    
    # 2. HD Gauss
    df_hd = run_scenario("HD_Gauss", lambda **k: make_toy_hd_gauss(n_features=20, **k))
    if df_hd is not None: dfs.append(df_hd)
    
    # 3. Adult
    df_adult = run_scenario("Adult", load_adult_income)
    if df_adult is not None: dfs.append(df_adult)
    
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        os.makedirs("experiments/results", exist_ok=True)
        full_df.to_csv("experiments/results/benchmarks.csv", index=False)
        print("\nResults saved to experiments/results/benchmarks.csv")
        print(full_df.groupby("scenario").mean(numeric_only=True))

if __name__ == "__main__":
    main()
