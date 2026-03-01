import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute K×K confusion matrix with normalization modes:
    'none', 'true', 'pred', 'all'
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Handle empty input
    if y_true.size == 0:
        if num_classes is None:
            return np.zeros((0, 0), dtype=int)
        return np.zeros((num_classes, num_classes), dtype=int)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have same shape")

    # Infer number of classes
    if num_classes is None:
        num_classes = int(max(y_true.max(), y_pred.max())) + 1

    # Validate label range
    if (y_true.min() < 0 or y_pred.min() < 0 or
        y_true.max() >= num_classes or y_pred.max() >= num_classes):
        raise ValueError("Labels must be in range [0, num_classes-1]")

    K = num_classes

    # Vectorized bincount trick
    indices = y_true * K + y_pred
    cm = np.bincount(indices, minlength=K*K).reshape(K, K)

    if normalize == 'none':
        return cm.astype(int)

    cm = cm.astype(float)
    eps = 1e-12

    if normalize == 'true':      # Row normalization
        row_sums = cm.sum(axis=1, keepdims=True)
        return cm / (row_sums + eps)

    elif normalize == 'pred':    # Column normalization
        col_sums = cm.sum(axis=0, keepdims=True)
        return cm / (col_sums + eps)

    elif normalize == 'all':     # Total normalization
        total = cm.sum()
        return cm / (total + eps)

    else:
        raise ValueError("normalize must be one of: 'none', 'true', 'pred', 'all'")