from typing import Optional, Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, is_classifier
from sklearn.metrics import log_loss

from ..core import Array, StructuralFeatureMap

class DifficultyFeature(BaseEstimator, TransformerMixin):
    """
    Computes difficulty score based on base model loss or margin.
    
    If y is provided (e.g. training), computes per-sample loss.
    If y is missing (e.g. test), computes entropy or margin (uncertainty).
    
    For now, we implement a simplified version:
    - If base_model is a classifier, return predicted probability of correct class (negated log likelihood approx) or entropy.
    - If base_model is None, returns 0s (feature unavailable).
    """
    
    def fit_transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[Any] = None,
    ) -> Array:
        return self.fit(X, y, base_model).transform(X, y, base_model)
        
    def fit(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[Any] = None,
    ) -> "DifficultyFeature":
        return self

    def transform(
        self,
        X: Array,
        y: Optional[Array] = None,
        base_model: Optional[Any] = None,
    ) -> Array:
        if base_model is None:
            # Fallback if no model provided
            return np.zeros((X.shape[0], 1))
            
        # Check if model is fitted (has predict)
        if not hasattr(base_model, "predict"):
             return np.zeros((X.shape[0], 1))
             
        if is_classifier(base_model) and hasattr(base_model, "predict_proba"):
            probas = base_model.predict_proba(X)
            
            if y is not None:
                # Supervised difficulty: Cross-entropy loss
                # We want a scalar per sample
                # log_loss from sklearn is aggregate, so we implement manual
                
                # Assume y are indices or binary 0/1. 
                # If y is (n,), ensure int
                y_idx = y.astype(int)
                
                # Grab prob of true class
                # Handle binary vs multiclass
                if probas.shape[1] == 2:
                    # binary
                    p_true = probas[np.arange(len(y)), y_idx]
                else:
                    # multiclass
                    p_true = probas[np.arange(len(y)), y_idx]
                    
                # Loss = -log(p_true)
                eps = 1e-12
                difficulty = -np.log(np.maximum(p_true, eps))
                
            else:
                # Unsupervised difficulty: Uncertainty (Entropy)
                # H(p) = - sum p log p
                eps = 1e-12
                entropy = -np.sum(probas * np.log(np.maximum(probas, eps)), axis=1)
                difficulty = entropy
                
        else:
            # Regressor or classifier without proba
            pred = base_model.predict(X)
            if y is not None:
                # Squared error
                difficulty = (y - pred) ** 2
            else:
                # Without y, hard to define difficulty for regression without variance estimate
                # Return 0s or perhaps magnitude of prediction?
                # Let's return 0s for safety
                difficulty = np.zeros(X.shape[0])
                
        return difficulty.reshape(-1, 1)

