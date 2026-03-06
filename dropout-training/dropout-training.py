import numpy as np

def dropout(x, p=0.5, rng=None):
    x = np.array(x)

    if p == 0.0:
        return x, np.ones_like(x)

    scale = 1 / (1 - p)

    # choose random generator
    if rng is not None:
        rand = np.random.default_rng(rng).random(size=x.shape)
    else:
        rand = np.random.random(size=x.shape)

    keep = rand < (1 - p)

    dropout_pattern = np.where(keep, scale, 0)

    output = x * dropout_pattern

    return (output, dropout_pattern)