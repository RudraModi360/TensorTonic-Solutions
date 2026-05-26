import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    return gamma*((x-np.mean(x,keepdims=True,axis=-1))/(np.sqrt((np.std(x,keepdims=True,axis=-1)**2+eps))))+beta