import numpy as np

def gini_impurity(y_left, y_right):
    y_left = np.array(y_left)
    y_right = np.array(y_right)

    n_left = len(y_left)
    n_right = len(y_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    def compute_gini(y):
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    gini_left = compute_gini(y_left)
    gini_right = compute_gini(y_right)

    gini_split = (n_left * gini_left + n_right * gini_right) / n_total

    return float(gini_split)