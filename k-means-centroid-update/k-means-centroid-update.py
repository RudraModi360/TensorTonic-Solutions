import numpy as np

def k_means_centroid_update(points, assignments, k):
    points = np.array(points, dtype=float)
    assignments = np.array(assignments)
    
    n_features = points.shape[1]
    centroids = []
    
    for i in range(k):
        cluster_points = points[assignments == i]
        
        if len(cluster_points) > 0:
            centroid = np.mean(cluster_points, axis=0)
        else:
            centroid = np.zeros(n_features)
        
        centroids.append(centroid.tolist())  # <-- IMPORTANT
    
    return centroids