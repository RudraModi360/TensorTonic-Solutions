import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    
    Parameters:
        T: (4,4) transformation matrix
        points: (3,) or (N,3)
    
    Returns:
        transformed points in same shape as input
    """
    T = np.array(T)
    points = np.array(points)

    # Ensure shape (N, 3)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    N = points.shape[0]

    # Convert to homogeneous (N,4)
    ones = np.ones((N, 1))
    points_h = np.hstack((points, ones))

    # Apply transformation
    transformed_h = (T @ points_h.T).T  # (N,4)

    # Convert back to 3D
    transformed = transformed_h[:, :3]

    # Return original shape
    return transformed[0] if N == 1 else transformed