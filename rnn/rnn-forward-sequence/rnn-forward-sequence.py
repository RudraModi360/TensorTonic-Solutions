import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """

    batch_size, T, input_dim = X.shape

    hidden_states = []

    h_prev = h_0

    for t in range(T):

        # Correct timestep extraction
        x_t = X[:, t, :]

        # Hidden state update
        h_t = np.tanh(
            np.dot(x_t, W_xh.T) +
            np.dot(h_prev, W_hh.T) +
            b_h
        )

        hidden_states.append(h_t)

        h_prev = h_t

    # Convert list -> array
    hidden_states = np.stack(hidden_states, axis=1)

    return hidden_states, h_t