import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    v = np.array(v)
    w = np.array(w)

    dot = np.dot(v, w)
    norm_v = np.linalg.norm(v)
    norm_w = np.linalg.norm(w)

    cos_theta = dot / (norm_v * norm_w)

    # Numerical stability
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    return np.arccos(cos_theta)