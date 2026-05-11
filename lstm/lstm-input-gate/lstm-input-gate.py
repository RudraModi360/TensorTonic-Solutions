import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray):
    """
    Compute:
        i_t = sigmoid(W_i @ [h,x] + b_i)
        c_hat = tanh(W_c @ [h,x] + b_c)
    """

    combined = np.concatenate((h_prev, x_t), axis=-1)

    i_t = sigmoid(combined @ W_i.T + b_i)

    c_hat = np.tanh(combined @ W_c.T + b_c)

    return i_t, c_hat