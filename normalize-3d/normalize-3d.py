import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    Works for shape (3,) or (N, 3)
    """
    v = np.array(v)

    norm = np.linalg.norm(v, axis=-1, keepdims=True)

    # Avoid division by zero
    norm = np.where(norm == 0, 1, norm)

    return v / norm