import numpy as np

def impute_missing(X, strategy='mean'):
    """
    Impute missing values (np.nan) using column-wise mean or median.
    """
    X = np.array(X, dtype=float)  # upcast to float
    result = X.copy()

    # Handle 1D case
    if result.ndim == 1:
        mask = ~np.isnan(result)
        
        if np.any(mask):
            if strategy == 'mean':
                fill_value = np.mean(result[mask])
            elif strategy == 'median':
                fill_value = np.median(result[mask])
            else:
                raise ValueError("strategy must be 'mean' or 'median'")
        else:
            fill_value = 0.0  # all NaN case
        
        result[~mask] = fill_value
        return result

    # Handle 2D case
    N, D = result.shape

    for col in range(D):
        column = result[:, col]
        mask = ~np.isnan(column)

        if np.any(mask):
            if strategy == 'mean':
                fill_value = np.mean(column[mask])
            elif strategy == 'median':
                fill_value = np.median(column[mask])
            else:
                raise ValueError("strategy must be 'mean' or 'median'")
        else:
            fill_value = 0.0  # all NaN column

        column[~mask] = fill_value
        result[:, col] = column

    return result