import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    x=np.array(x)
    p=np.array(p)
    if np.allclose(np.sum(p),1):
        return np.sum(x*p)
    else:
        raise ValueError