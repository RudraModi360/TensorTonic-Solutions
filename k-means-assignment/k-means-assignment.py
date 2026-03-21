import numpy as np

def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    points = np.array(points)        # shape: (n_samples, n_features)
    centroids = np.array(centroids)  # shape: (k, n_features)
    
    # Compute squared Euclidean distances
    # Result shape: (n_samples, k)
    distances = np.sum((points[:, np.newaxis] - centroids) ** 2, axis=2)
    
    # Assign each point to closest centroid
    return np.argmin(distances, axis=1).tolist()
    