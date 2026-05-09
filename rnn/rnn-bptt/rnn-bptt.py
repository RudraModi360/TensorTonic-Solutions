import numpy as np

def bptt_single_step(dh_next: np.ndarray, h_t: np.ndarray, h_prev: np.ndarray,
                     x_t: np.ndarray, W_hh: np.ndarray) -> tuple:
    """
    Backprop through one RNN time step.
    Returns (dh_prev, dW_hh).
    """
    tanh_derivation=1-h_t**2
    d_tanh=tanh_derivation*dh_next
    d_w_hh=(d_tanh.T)@h_prev
    dh_prev=d_tanh@W_hh
    return dh_prev,d_w_hh
    