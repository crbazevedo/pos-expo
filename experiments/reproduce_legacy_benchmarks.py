"""
Reproduction script for legacy benchmarks.
Compares results from 'legacy' scripts (in examples/) against a fresh library-based implementation.

Usage:
    python experiments/reproduce_legacy_benchmarks.py
"""

import sys
import os
import subprocess
import re
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Any, Optional

# Add src to path to ensure we use local library for _lib functions
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pos_expo.datasets import make_toy_selection_bias, load_adult_income
from pos_expo.features import build_default_feature_map
from pos_expo.estimators import PosExpoClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import time

def run_script_and_capture_output(script_path: str) -> Dict[str, Any]:
    """Runs a python script and returns stdout and runtime."""
    start_time = time.time()
    cmd = [sys.executable, script_path]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    runtime = time.time() - start_time
    return {"stdout": result.stdout, "runtime": runtime}

# --- 1D Piecewise ---

def run_1d_piecewise_legacy() -> Dict[str, float]:
    script_path = os.path.join("examples", "toy_selection_bias.py")
    print(f"Running legacy script: {script_path}")
    res = run_script_and_capture_output(script_path)
    stdout = res["stdout"]
    runtime = res["runtime"]
    
    # Parse output
    erm_match = re.search(r"ERM Test Log Loss:\s*([\d\.]+)", stdout)
    pos_match = re.search(r"POS-Expo Test Log Loss:\s*([\d\.]+)", stdout)
    
    if not erm_match or not pos_match:
        raise ValueError(f"Could not parse metrics from {script_path}. Output:\n{stdout}")
        
    return {
        "ERM": float(erm_match.group(1)),
        "POS-Expo": float(pos_match.group(1)),
        "runtime": runtime
    }

def run_1d_piecewise_lib() -> Dict[str, float]:
    print("Running library implementation for 1D Piecewise")
    start_time = time.time()
    # Match parameters from examples/toy_selection_bias.py
    n_samples = 2000
    random_state = 42
    
    X_train, y_train, X_test, y_test, _ = make_toy_selection_bias(n_samples=n_samples, random_state=random_state)
    X_ref = X_test
    
    # ERM
    erm = LogisticRegression()
    erm.fit(X_train, y_train)
    loss_erm = log_loss(y_test, erm.predict_proba(X_test))
    
    # POS-Expo
    feature_map = build_default_feature_map()
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(),
        feature_map=feature_map,
        alpha_reg=0.1
    )
    clf.fit(X_train, y_train, X_ref=X_ref)
    loss_pos = log_loss(y_test, clf.predict_proba(X_test))
    
    runtime = time.time() - start_time
    return {
        "ERM": loss_erm,
        "POS-Expo": loss_pos,
        "runtime": runtime
    }

# --- Adult ---

def run_adult_legacy() -> Dict[str, float]:
    script_path = os.path.join("examples", "adult_income_pos_expo.py")
    print(f"Running legacy script: {script_path}")
    res = run_script_and_capture_output(script_path)
    stdout = res["stdout"]
    runtime = res["runtime"]
    
    erm_match = re.search(r"ERM .* Log Loss:\s*([\d\.]+)", stdout)
    pos_match = re.search(r"POS-Expo .* Log Loss:\s*([\d\.]+)", stdout)
    
    if not erm_match or not pos_match:
        raise ValueError(f"Could not parse metrics from {script_path}. Output:\n{stdout}")
        
    return {
        "ERM": float(erm_match.group(1)),
        "POS-Expo": float(pos_match.group(1)),
        "runtime": runtime
    }

def run_adult_lib() -> Dict[str, float]:
    print("Running library implementation for Adult")
    start_time = time.time()
    # Match parameters from examples/adult_income_pos_expo.py
    
    try:
        X_train, y_train, X_test, y_test, X_ref = load_adult_income(biased=True, random_state=42)
    except Exception as e:
        warnings.warn(f"Failed to load Adult data: {e}. Skipping lib run.")
        return {"ERM": np.nan, "POS-Expo": np.nan, "runtime": np.nan}

    if X_ref is None:
        X_ref = X_test
        
    # ERM
    erm = LogisticRegression(max_iter=1000)
    erm.fit(X_train, y_train)
    loss_erm = log_loss(y_test, erm.predict_proba(X_test))
    
    # POS-Expo
    feature_map = build_default_feature_map(rarity_k=20, diversity_clusters=20)
    clf = PosExpoClassifier(
        base_estimator=LogisticRegression(max_iter=1000),
        feature_map=feature_map,
        alpha_reg=1.0
    )
    clf.fit(X_train, y_train, X_ref=X_ref)
    loss_pos = log_loss(y_test, clf.predict_proba(X_test))
    
    runtime = time.time() - start_time
    return {
        "ERM": loss_erm,
        "POS-Expo": loss_pos,
        "runtime": runtime
    }

# --- Comparison ---

def compare_and_log(scenario: str, legacy: Dict[str, float], lib: Dict[str, float]) -> pd.DataFrame:
    rows = []
    models = ["ERM", "POS-Expo"]
    
    print(f"\n--- Comparison: {scenario} ---")
    print(f"Runtime: Legacy={legacy.get('runtime', 0):.4f}s, Lib={lib.get('runtime', 0):.4f}s")
    
    for model in models:
        l_val = legacy.get(model, np.nan)
        lib_val = lib.get(model, np.nan)
        diff = abs(l_val - lib_val) if not np.isnan(l_val) and not np.isnan(lib_val) else np.nan
        
        print(f"{model}: Legacy={l_val:.6f}, Lib={lib_val:.6f}, Diff={diff:.6f}")
        
        if diff > 1e-4: # Tolerance
             warnings.warn(f"Mismatch in {scenario} for {model}: {diff} > 1e-4")
             
        rows.append({
            "scenario": scenario,
            "model": model,
            "test_loss_legacy": l_val,
            "test_loss_lib": lib_val,
            "test_loss_diff": diff,
            "runtime_legacy": legacy.get('runtime'),
            "runtime_lib": lib.get('runtime')
        })
    return pd.DataFrame(rows)

def main():
    all_results = []
    
    # 1D Piecewise
    try:
        leg_1d = run_1d_piecewise_legacy()
        lib_1d = run_1d_piecewise_lib()
        df_1d = compare_and_log("1D_Piecewise", leg_1d, lib_1d)
        all_results.append(df_1d)
    except Exception as e:
        print(f"Error in 1D Piecewise: {e}")
        import traceback
        traceback.print_exc()

    # Adult
    try:
        leg_adult = run_adult_legacy()
        lib_adult = run_adult_lib()
        df_adult = compare_and_log("Adult", leg_adult, lib_adult)
        all_results.append(df_adult)
    except Exception as e:
        print(f"Error in Adult: {e}")
        import traceback
        traceback.print_exc()
        
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_path = os.path.join(RESULTS_DIR, "legacy_vs_lib.csv")
        final_df.to_csv(out_path, index=False)
        print(f"\nSaved comparison results to {out_path}")
        print(final_df)

if __name__ == "__main__":
    main()

