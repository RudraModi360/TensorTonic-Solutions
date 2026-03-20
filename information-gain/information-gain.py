import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    y = np.asarray(y)
    split_mask = np.asarray(split_mask, dtype=bool)

    # Split
    y_left = y[split_mask]
    y_right = y[~split_mask]

    if y_left is None or y_right is None:
        return 0.0

    # Weights
    n = len(y)
    w_left = len(y_left) / n
    w_right = len(y_right) / n

    # IG formula
    ig = _entropy(y) - (w_left * _entropy(y_left) + w_right * _entropy(y_right))

    return ig