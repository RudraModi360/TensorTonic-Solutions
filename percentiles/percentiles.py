import numpy as np

def percentiles(x, q):
    """
    Compute percentiles using linear interpolation.
    """
    x=np.array(x)
    q=np.array(q)
    return np.percentile(x,q,method='linear')