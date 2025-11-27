import numpy as np
from pos_expo.poset import compute_poset_layers

def test_poset_layers_chain():
    # Simple chain: (0), (1), (2)
    phi = np.array([[0], [1], [2]])
    layers = compute_poset_layers(phi)
    
    assert np.array_equal(layers, [0, 1, 2])

def test_poset_layers_branch():
    # (0,0) -> (1,0)
    # (0,0) -> (0,1)
    # (1,0) -> (1,1)
    # (0,1) -> (1,1)
    
    # Order: a < b, a < c, b < d, c < d
    # Layers: a=0, b=1, c=1, d=2
    
    phi = np.array([
        [0, 0], # a
        [1, 0], # b
        [0, 1], # c
        [1, 1], # d
    ])
    
    layers = compute_poset_layers(phi)
    
    # Since ordering of output depends on input order
    # a is index 0, b=1, c=2, d=3
    
    assert layers[0] == 0
    assert layers[1] == 1
    assert layers[2] == 1
    assert layers[3] == 2

