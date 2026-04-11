import numpy as np

def one_hot(y, num_classes=None):
    """
    Convert integer labels y ∈ {0,...,K-1} into one-hot matrix of shape (N, K).
    """
    y = np.array(y, dtype=int)

    if num_classes is None:
        num_classes = np.max(y) + 1  # safer than len(set(y))

    y_one_hot = np.zeros((len(y), num_classes), dtype=int)
    y_one_hot[np.arange(len(y)), y] = 1

    return y_one_hot