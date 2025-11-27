from typing import Dict, Any, Optional
import numpy as np

from .core import Array

def weight_statistics(weights: Array) -> Dict[str, float]:
    """
    Computes summary statistics of the weights.
    """
    weights = np.asarray(weights)
    n = len(weights)
    
    mean_w = np.mean(weights)
    std_w = np.std(weights)
    min_w = np.min(weights)
    max_w = np.max(weights)
    
    # Effective Sample Size (ESS)
    # ESS = (sum w)^2 / sum w^2
    ess = (np.sum(weights) ** 2) / np.sum(weights ** 2)
    
    return {
        "mean": float(mean_w),
        "std": float(std_w),
        "min": float(min_w),
        "max": float(max_w),
        "ess": float(ess),
        "ess_ratio": float(ess / n),
    }

def structural_moments(
    weights: Array,
    phi: Array,
) -> Dict[str, Array]:
    """
    Computes weighted moments of the structural features.
    
    Args:
        weights: (n_samples,)
        phi: (n_samples, k)
        
    Returns:
        Dict with "mean" (k,) and "std" (k,) under weighted distribution.
    """
    weights = np.asarray(weights)
    # Normalize weights for moment computation
    p = weights / np.sum(weights)
    
    # Weighted mean: E[phi]
    mean_phi = np.average(phi, axis=0, weights=p)
    
    # Weighted variance: E[(phi - mean)^2]
    # var = sum p_i (phi_i - mean)^2
    var_phi = np.average((phi - mean_phi) ** 2, axis=0, weights=p)
    std_phi = np.sqrt(var_phi)
    
    return {
        "mean": mean_phi,
        "std": std_phi,
    }

def covariate_shift_summary(
    X_train: Array,
    X_ref: Array,
    weights: Array,
    feature_map: Any,
) -> Dict[str, Any]:
    """
    Summarizes how POS-Expo changed structural moments compared to unweighted.
    Also compares with Reference moments.
    """
    # 1. Compute phi
    phi_train = feature_map.transform(X_train)
    phi_ref = feature_map.transform(X_ref)
    
    # 2. Moments
    # Train (Unweighted)
    m_train_unweighted = structural_moments(np.ones(len(X_train)), phi_train)
    
    # Train (Weighted)
    m_train_weighted = structural_moments(weights, phi_train)
    
    # Ref (Target)
    m_ref = structural_moments(np.ones(len(X_ref)), phi_ref)
    
    return {
        "train_unweighted": m_train_unweighted,
        "train_weighted": m_train_weighted,
        "ref": m_ref,
        "shift_correction_l2": np.linalg.norm(m_train_weighted["mean"] - m_ref["mean"]),
        "original_shift_l2": np.linalg.norm(m_train_unweighted["mean"] - m_ref["mean"]),
    }

