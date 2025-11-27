from typing import Tuple
import numpy as np
from scipy.stats import norm

from ..core import Array

def make_toy_selection_bias(
    n_samples: int = 1000,
    random_state: int = 42,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Generates a 1D synthetic dataset with selection bias.
    
    P_train(x) is biased towards positive x.
    P_test(x) is standard normal.
    
    Returns:
        X_train, y_train, X_test, y_test, iw_star (for X_train)
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Define P_test (Target) as N(0, 1)
    # We sample test data directly from P_test
    n_test = n_samples # same size for simplicity
    X_test = rng.normal(0, 1, size=(n_test, 1))
    
    # 2. Define P_train (Source)
    # We want selection bias. Let's say we sample from N(1, 1).
    # Then w*(x) = P_test(x) / P_train(x) = exp(-0.5*x^2) / exp(-0.5*(x-1)^2) * C
    #             = exp( -0.5*x^2 + 0.5*(x^2 - 2x + 1) ) * C
    #             = exp( -x + 0.5 ) * C
    # Basically, weights decrease as x increases.
    
    X_train = rng.normal(1, 1, size=(n_samples, 1))
    
    # Calculate true importance weights for X_train samples
    # pdf_test = N(0,1), pdf_train = N(1,1)
    pdf_test = norm.pdf(X_train, loc=0, scale=1)
    pdf_train = norm.pdf(X_train, loc=1, scale=1)
    iw_star = (pdf_test / pdf_train).ravel()
    
    # 3. Define y|x (concept) - same for train and test
    # y = sign(x) with some noise (linear decision boundary at x=0)
    # P(y=1|x) = sigmoid(beta * x)
    def generate_y(X):
        logits = 2.0 * X[:, 0] # Steep sigmoid
        probs = 1 / (1 + np.exp(-logits))
        y = (rng.rand(len(X)) < probs).astype(int)
        return y
        
    y_train = generate_y(X_train)
    y_test = generate_y(X_test)
    
    return X_train, y_train, X_test, y_test, iw_star

def make_toy_hd_gauss(
    n_samples: int = 1000,
    n_features: int = 10,
    random_state: int = 42,
) -> Tuple[Array, Array, Array, Array, Array]:
    """
    Generates a higher-dimensional Gaussian dataset with selection bias.
    
    P_test = N(0, I)
    P_train = N(mu, I) where mu = [1, 1, ..., 0, 0] (bias in first 2 dims)
    """
    rng = np.random.RandomState(random_state)
    
    # Target: N(0, I)
    X_test = rng.normal(0, 1, size=(n_samples, n_features))
    
    # Source: N(mu, I)
    mu = np.zeros(n_features)
    mu[:2] = 1.0 # shift in first two dimensions
    
    X_train = rng.normal(0, 1, size=(n_samples, n_features)) + mu
    
    # Importance weights
    # w(x) = p_test(x) / p_train(x)
    # log w = -0.5 x^2 + 0.5 (x-mu)^2 = -0.5 x^2 + 0.5(x^2 - 2x.mu + mu^2)
    #       = -x.mu + 0.5 mu^2
    # So weights depend exponentially on projection onto mu.
    
    log_iw = - X_train @ mu + 0.5 * np.sum(mu**2)
    iw_star = np.exp(log_iw)
    
    # Labels y
    # y depends on X[:, 0] and X[:, 2] (one biased, one unbiased feature)
    def generate_y(X):
        # Linear boundary
        logits = 2.0 * X[:, 0] - 2.0 * X[:, 2]
        probs = 1 / (1 + np.exp(-logits))
        y = (rng.rand(len(X)) < probs).astype(int)
        return y
        
    y_train = generate_y(X_train)
    y_test = generate_y(X_test)
    
    return X_train, y_train, X_test, y_test, iw_star
