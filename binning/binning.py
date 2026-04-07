import numpy as np

def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    Returns bin indices from 0 to num_bins-1
    """
    values = np.array(values)

    min_val = np.min(values)
    max_val = np.max(values)

    # Avoid division by zero
    if min_val == max_val:
        return np.zeros_like(values, dtype=int).tolist()

    bin_width = (max_val - min_val) / num_bins

    bins = ((values - min_val) / bin_width).astype(int)

    # Fix edge case where value == max_val
    bins = np.clip(bins, 0, num_bins - 1)

    return bins.tolist()