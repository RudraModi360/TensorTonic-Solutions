import numpy as np

def gini(y):
    """Compute Gini impurity of labels."""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)


def decision_tree_split(X, y):
    """
    Find the best (feature, threshold) maximizing information gain.
    
    Returns:
        best_feature, best_threshold, best_gain
    """
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    parent_gini = gini(y)
    
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    for feature in range(n_features):
        values = X[:, feature]
        
        # Sort values and corresponding labels
        sorted_idx = np.argsort(values)
        values_sorted = values[sorted_idx]
        y_sorted = y[sorted_idx]
        
        # Unique split candidates (midpoints)
        unique_vals = np.unique(values_sorted)
        if len(unique_vals) == 1:
            continue
        
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2
        
        for threshold in thresholds:
            left_mask = values <= threshold
            right_mask = values > threshold
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # Skip invalid splits
            if len(y_left) == 0 or len(y_right) == 0:
                continue
            
            # Compute weighted Gini
            gini_left = gini(y_left)
            gini_right = gini(y_right)
            
            w_left = len(y_left) / n_samples
            w_right = len(y_right) / n_samples
            
            gini_split = w_left * gini_left + w_right * gini_right
            
            # Information gain
            gain = parent_gini - gini_split
            
            # Tie-breaking rules
            if (gain > best_gain or
                (gain == best_gain and feature < best_feature) or
                (gain == best_gain and feature == best_feature and threshold < best_threshold)):
                
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    return best_feature, best_threshold