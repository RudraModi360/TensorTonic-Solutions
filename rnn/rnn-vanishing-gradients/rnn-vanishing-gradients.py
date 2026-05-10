import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    """
    Simulate gradient norm decay over T time steps.
    Returns list of gradient norms.
    """
    grad_norm=np.linalg.norm(W_hh,ord=2)
    gradient=[1.0]
    current=1.0

    for time in range(1,T):
        current*=grad_norm
        gradient.append(float(current))
    return gradient
    