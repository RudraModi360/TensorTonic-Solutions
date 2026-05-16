import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class GRU:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):

        self.hidden_dim = hidden_dim

        scale = np.sqrt(2.0 / (input_dim + hidden_dim))

        # GRU weights
        self.W_r = np.random.randn(
            hidden_dim,
            hidden_dim + input_dim
        ) * scale

        self.W_z = np.random.randn(
            hidden_dim,
            hidden_dim + input_dim
        ) * scale

        self.W_h = np.random.randn(
            hidden_dim,
            hidden_dim + input_dim
        ) * scale

        # Biases
        self.b_r = np.zeros(hidden_dim)
        self.b_z = np.zeros(hidden_dim)
        self.b_h = np.zeros(hidden_dim)

        # Output layer
        self.W_y = np.random.randn(
            output_dim,
            hidden_dim
        ) * np.sqrt(2.0 / (hidden_dim + output_dim))

        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        """
        X shape:
            (batch_size, seq_len, input_dim)

        Returns:
            y       -> (batch_size, seq_len, output_dim)
            h_last  -> (batch_size, hidden_dim)
        """

        batch_size, seq_len, _ = X.shape

        # Initial hidden state
        h_prev = np.zeros((batch_size, self.hidden_dim))

        outputs = []

        for t in range(seq_len):

            # Current timestep input
            x_t = X[:, t, :]

            # Concatenate hidden + input
            combined = np.concatenate(
                (h_prev, x_t),
                axis=-1
            )

            # Reset gate
            r_t = sigmoid(
                combined @ self.W_r.T + self.b_r
            )

            # Update gate
            z_t = sigmoid(
                combined @ self.W_z.T + self.b_z
            )

            # Candidate hidden state
            combined_candidate = np.concatenate(
                (r_t * h_prev, x_t),
                axis=-1
            )

            h_tilde = np.tanh(
                combined_candidate @ self.W_h.T + self.b_h
            )

            # Final hidden state
            h_t = (
                z_t * h_prev
                + (1-z_t) * h_tilde
            )

            # Output
            y_t = h_t @ self.W_y.T + self.b_y

            outputs.append(y_t)

            # Update hidden state
            h_prev = h_t

        # Stack outputs across sequence dimension
        y = np.stack(outputs, axis=1)

        return y, h_prev