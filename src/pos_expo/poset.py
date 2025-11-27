import numpy as np
import warnings
from .core import Array

def compute_poset_layers(phi: Array) -> Array:
    """
    Computes approximate poset layers for the point set phi.
    
    Layer 0: Minimal elements (no z' such that z' < z).
    Layer k: Elements whose longest chain of predecessors has length k.
    
    Warning: O(N^2). Use only for diagnostics on small subsamples.
    
    Args:
        phi: (n_samples, k_features)
        
    Returns:
        layers: (n_samples,) int array of layer indices.
    """
    n = phi.shape[0]
    
    # Simple iterative approach
    # Ideally we'd build a DAG, but O(N^2) comparison is the bottleneck anyway.
    
    if n > 2000:
        warnings.warn(f"compute_poset_layers called with N={n}, this will be slow.", category=UserWarning)
        
    # Re-implementation with sort
    # Topological sort first: Sorting by sum of coordinates is a topological sort for product order!
    scores = phi.sum(axis=1)
    perm = np.argsort(scores)
    inv_perm = np.argsort(perm)
    
    phi_sorted = phi[perm]
    layers_sorted = np.zeros(n, dtype=int)
    
    # Iterate through sorted elements. For each j, look at all i < j (in index)
    # Check if phi_sorted[i] < phi_sorted[j]
    
    # This is still O(N^2) but only checks "past" elements.
    for j in range(n):
        # Candidates i are 0..j-1
        if j == 0:
            continue
            
        # We need to find max(layers[i]) for i < j such that phi[i] < phi[j]
        # Vectorized check against block 0..j-1
        current_phi = phi_sorted[j]
        past_phis = phi_sorted[:j]
        
        # Check strict dominance: all coords <= and not all equal
        # Since sum is strictly increasing (or equal), and we process in order,
        # we just check <=.
        # Wait, sum(i) <= sum(j) is necessary but not sufficient.
        
        is_pred = np.all(past_phis <= current_phi, axis=1)
        # Exclude equality if it happens (though sort order usually handles it unless duplicate)
        is_equal = np.all(past_phis == current_phi, axis=1)
        is_strict_pred = is_pred & (~is_equal)
        
        if np.any(is_strict_pred):
            layers_sorted[j] = 1 + np.max(layers_sorted[:j][is_strict_pred])
        else:
            layers_sorted[j] = 0
            
    # Map back to original indices
    layers = layers_sorted[inv_perm]
    return layers
