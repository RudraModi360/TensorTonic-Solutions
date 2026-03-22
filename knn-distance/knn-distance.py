import numpy as np

def knn_distance(X_train, X_test, k):
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # Ensure 2D
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Compute distances
    distances = np.sqrt(
        np.sum((X_test[:, None, :] - X_train[None, :, :]) ** 2, axis=2)
    )

    # Sort indices
    sorted_idx = np.argsort(distances, axis=1)

    n_train = X_train.shape[0]

    # Prepare output with -1 padding
    result = -1 * np.ones((X_test.shape[0], k), dtype=int)

    # Fill available neighbors
    valid_k = min(k, n_train)
    result[:, :valid_k] = sorted_idx[:, :valid_k]

    return result