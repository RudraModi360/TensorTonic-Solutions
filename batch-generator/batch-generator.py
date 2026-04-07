import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Yield shuffled mini-batches of (X_batch, y_batch).

    Parameters:
    X : array-like, shape (N, ...) 
    y : array-like, shape (N,)
    batch_size : int
    rng : np.random.Generator (optional)
    drop_last : bool

    Yields:
    (X_batch, y_batch)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    N = len(X)
    assert len(y) == N, "X and y must have same length"

    # Use provided RNG or default
    if rng is None:
        rng = np.random.default_rng()

    # Generate a single permutation
    indices = rng.permutation(N)

    # Shuffle without modifying original arrays
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Yield batches
    for start in range(0, N, batch_size):
        end = start + batch_size

        # Handle drop_last
        if drop_last and end > N:
            break

        yield X_shuffled[start:end], y_shuffled[start:end]