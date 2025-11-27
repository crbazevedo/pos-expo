import numpy as np
import pytest
from pos_expo.datasets import load_adult_income, make_toy_hd_gauss, make_toy_selection_bias

def test_load_adult_income_dummy():
    try:
        X_train, y_train, X_test, y_test, X_ref = load_adult_income(biased=True, download_if_missing=True)
        
        assert len(X_train) > 0
        assert len(y_train) == len(X_train)
        assert len(X_test) > 0
        
        if X_ref is not None:
             assert X_ref.shape[1] == X_train.shape[1]
             
    except Exception as e:
        pytest.fail(f"Adult loader failed: {e}")

def test_make_toy_hd_gauss():
    X_train, y_train, X_test, y_test, iw_star = make_toy_hd_gauss(n_samples=100, n_features=5)
    
    assert X_train.shape == (100, 5)
    assert X_test.shape == (100, 5)
    assert iw_star.shape == (100,)
    assert np.all(iw_star > 0)

def test_make_toy_selection_bias():
    X_train, y_train, X_test, y_test, iw_star = make_toy_selection_bias(n_samples=100)
    assert X_train.shape == (100, 1)
