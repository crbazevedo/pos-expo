import numpy as np
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
    
    # Dominance matrix: D[i, j] = 1 if phi[i] <= phi[j] (and i != j)
    # Actually strictly smaller on at least one dim, and <= on all?
    # Spec says: z <= z' iff phi_k(z) <= phi_k(z') for all k.
    
    # Let's define strictly dominated: i < j iff (phi[i] <= phi[j]).all() and (phi[i] != phi[j]).any()
    # But for layering, standard product order is reflexive.
    # Strict order needed for height.
    
    # 1. Compute strict dominance
    # diff[i, j, k] = phi[j, k] - phi[i, k]
    # This is O(N^2 * K) - heavy memory.
    
    # Let's do it loop-based (slower python, less memory) or broadcast chunks.
    # For small N (diagnostics), N=1000 is fine.
    
    layers = np.zeros(n, dtype=int)
    
    # Sort by sum(phi) to ensure we process roughly in order? 
    # Not strictly necessary for the definition, but helps some algos.
    # We will compute Longest Path in the DAG.
    
    # Adjacency: A[i, j] = 1 if i < j (strictly)
    # Using broadcasting for N <= 1000
    if n > 2000:
        # Fallback or warning
        print(f"Warning: compute_poset_layers called with N={n}, this will be slow.")
    
    # Expand dims
    # phi_i: (N, 1, K), phi_j: (1, N, K)
    # leq: (N, N, K) -> (N, N)
    phi_i = phi[:, np.newaxis, :]
    phi_j = phi[np.newaxis, :, :]
    
    is_leq = np.all(phi_i <= phi_j, axis=2) # i <= j
    is_eq = np.all(phi_i == phi_j, axis=2)  # i == j
    is_strict_less = is_leq & (~is_eq)      # i < j
    
    # Compute height: length of longest chain ending at j
    # H(j) = 1 + max_{i < j} H(i), with H(min) = 0
    
    # To solve this, we can iterate. 
    # Since it's a DAG, we can just repeatedly update until convergence?
    # Max depth is N.
    
    # Initialize layers to 0
    layers = np.zeros(n, dtype=int)
    
    for _ in range(n):
        prev_layers = layers.copy()
        
        # layer[j] = max(layer[i] for i where i < j) + 1
        # If no i < j, layer[j] = 0.
        
        # We can vectorize this update?
        # A[i, j] is 1 if i < j.
        # We want max over i of (A[i, j] * (layers[i] + 1))
        
        # Create matrix of potential values:
        # V[i, j] = layers[i] + 1 if i < j else 0
        
        # But this is O(N^2) inside loop. Total O(N^3).
        # A bit slow.
        
        # Optimization: Topological sort first?
        # Sorting by sum of coordinates is a topological sort for product order!
        # If x < y, then sum(x) < sum(y).
        break # We'll re-implement using the sort trick below.
        
    # Re-implementation with sort
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

