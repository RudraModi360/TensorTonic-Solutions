import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    
    Supports:
        (3,) single vector
        (N,3) multiple vectors
    """
    v = np.array(v)

    if v.ndim == 1:
        return np.sqrt(np.sum(v**2))
    else:
        return np.sqrt(np.sum(v**2, axis=1))