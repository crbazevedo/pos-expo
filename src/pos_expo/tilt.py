from typing import Optional, Dict, Any
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

from .core import StructuralFeatureMap, Array

def estimate_log_density_ratio(
    X_train: Array,
    X_ref: Array,
    logistic_kwargs: Optional[Dict[str, Any]] = None,
) -> Array:
    """
    Estimates log(dP_ref(x) / dP_train(x)) via logistic regression.
    
    Args:
        X_train: Training samples (domain 0).
        X_ref: Reference samples approximating target (domain 1).
        logistic_kwargs: Arguments for sklearn LogisticRegression.
        
    Returns:
        g_hat: Estimated log density ratio for X_train samples.
    """
    if logistic_kwargs is None:
        logistic_kwargs = {"C": 1.0, "max_iter": 1000}
        
    n_train = X_train.shape[0]
    n_ref = X_ref.shape[0]
    
    # Create dataset for discrimination
    X_concat = np.vstack([X_train, X_ref])
    y_concat = np.hstack([np.zeros(n_train), np.ones(n_ref)])
    
    clf = LogisticRegression(**logistic_kwargs)
    clf.fit(X_concat, y_concat)
    
    # Predict logits (log odds) on X_train
    # logit(p(d=1|x)) = log( p(x|d=1)p(d=1) / p(x|d=0)p(d=0) )
    #                 = log(p_ref/p_train) + log(n_ref/n_train)
    # So log(p_ref/p_train) = logit(x) - log(n_ref/n_train)
    
    # decision_function returns the logit (signed distance to hyperplane)
    logits = clf.decision_function(X_train)
    
    # Adjust for class imbalance
    log_prior_ratio = np.log(n_ref / n_train)
    log_density_ratio = logits - log_prior_ratio
    
    return log_density_ratio

class PosExpoReweighter:
    """
    d q_alpha / dP_train (z) propto exp(<alpha, phi(z)>).
    Learns alpha by projecting log-density ratio onto span{phi}.
    """

    def __init__(
        self,
        feature_map: StructuralFeatureMap,
        alpha_reg: float = 1e-3,
        allow_negative_alpha: bool = True,
    ) -> None:
        self.feature_map = feature_map
        self.alpha_reg = alpha_reg
        self.allow_negative_alpha = allow_negative_alpha
        
        self.alpha_: Optional[Array] = None
        self.bias_: float = 0.0

    def fit(
        self,
        X_train: Array,
        y_train: Optional[Array] = None,
        X_ref: Optional[Array] = None,
        iw_star: Optional[Array] = None,
    ) -> "PosExpoReweighter":
        """
        Fit the reweighter.
        
        If iw_star is provided:
            Performs L2 regression of log(iw_star) on phi(z).
            (I-projection of the oracle importance weights onto the exponential family).
        
        If iw_star is None and X_ref is provided:
            Estimates log density ratio g(z) via logistic regression (train vs ref)
            and then projects g onto span{phi}.
        """
        # Compute structural features on training data
        phi_train = self.feature_map.fit_transform(X_train, y_train)

        target_log_w = None

        if iw_star is not None:
            # Case 1: Oracle weights provided
            eps = 1e-12
            target_log_w = np.log(np.maximum(iw_star, eps))
            
        elif X_ref is not None:
            # Case 2: Estimate from X_ref
            # We need to estimate g_hat on X_train
            # Note: The feature map might need to be fit on X_train already (done above).
            # But estimate_log_density_ratio uses raw features (X) to find the ratio first.
            target_log_w = estimate_log_density_ratio(X_train, X_ref)
            
        else:
            raise ValueError("Either iw_star or X_ref must be provided.")

        # Fit (bias, alpha) to target_log_w
        ridge = Ridge(alpha=self.alpha_reg, fit_intercept=True)
        ridge.fit(phi_train, target_log_w)
        
        self.alpha_ = ridge.coef_
        self.bias_ = ridge.intercept_
        
        if not self.allow_negative_alpha:
            # Simple projection onto non-negative orthant if requested
            self.alpha_ = np.maximum(self.alpha_, 0.0)
            
        return self

    def compute_weights(
        self,
        X: Array,
        y: Optional[Array] = None,
        normalize: bool = True,
    ) -> Array:
        """
        Compute w_alpha(z) = exp(bias + alpha^T phi(z)).
        """
        if self.alpha_ is None:
            raise RuntimeError("PosExpoReweighter is not fitted.")

        phi = self.feature_map.transform(X, y)
        
        # log_w = bias + alpha . phi
        log_w = self.bias_ + phi @ self.alpha_
        
        weights = np.exp(log_w)
        
        if normalize:
            # Normalize so mean weight is 1 (preserves effective sample size scale)
            weights /= np.mean(weights)
            
        return weights
