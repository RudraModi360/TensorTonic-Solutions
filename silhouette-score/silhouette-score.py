import numpy as np

def silhouette_score(X, labels):
    X = np.asarray(X)
    labels = np.asarray(labels)

    n = X.shape[0]
    unique_labels = np.unique(labels)
    K = len(unique_labels)

    if K < 2:
        raise ValueError("Silhouette score requires at least 2 clusters")

    # -------- Pairwise Euclidean distance matrix --------
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff**2, axis=2))  # shape (n, n)

    # -------- Intra-cluster distance a(i) --------
    a = np.zeros(n)

    for label in unique_labels:
        mask = labels == label
        cluster_size = np.sum(mask)

        if cluster_size > 1:
            intra_dist = dist_matrix[mask][:, mask]
            # exclude diagonal
            a[mask] = (np.sum(intra_dist, axis=1) /
                       (cluster_size - 1))
        else:
            a[mask] = 0.0  # singleton cluster

    # -------- Inter-cluster distance b(i) --------
    b = np.full(n, np.inf)

    for label in unique_labels:
        mask_i = labels == label
        mask_other = labels != label

        for other_label in unique_labels:
            if other_label == label:
                continue

            mask_j = labels == other_label
            inter_dist = dist_matrix[mask_i][:, mask_j]
            mean_dist = np.mean(inter_dist, axis=1)

            b[mask_i] = np.minimum(b[mask_i], mean_dist)

    # -------- Silhouette per point --------
    s = (b - a) / np.maximum(a, b)

    # Handle edge case where a=b=0
    s[np.isnan(s)] = 0.0

    return np.mean(s)