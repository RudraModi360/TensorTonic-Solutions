import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    indices = np.arange(N)

    # Shuffle if needed
    if shuffle:
        rng = rng if rng is not None else np.random.default_rng()
        rng.shuffle(indices)

    # Compute fold sizes (handle uneven splits)
    fold_sizes = np.full(k, N // k)
    fold_sizes[:N % k] += 1  # distribute remainder

    folds = []
    current = 0

    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_idx = indices[start:end]
        train_idx = np.concatenate((indices[:start], indices[end:]))
        
        folds.append((train_idx, val_idx))
        current = end

    return folds