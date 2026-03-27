import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    x=np.array(x)
    mean=np.mean(x)
    median=np.median(x)
    val,counts=np.unique(x,return_counts=True)
    return mean, median, val[np.argmax(counts)]