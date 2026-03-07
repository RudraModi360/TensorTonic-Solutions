import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    
    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)

    if x.ndim == 2:  # (N,D)

        mean = np.mean(x, axis=0, keepdims=True)
        var = np.var(x, axis=0, keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        return gamma * x_hat + beta

    elif x.ndim == 4:  # (N,C,H,W)

        mean = np.mean(x, axis=(0,2,3), keepdims=True)
        var = np.var(x, axis=(0,2,3), keepdims=True)

        x_hat = (x - mean) / np.sqrt(var + eps)

        return gamma.reshape(1,-1,1,1) * x_hat + beta.reshape(1,-1,1,1)

    else:
        raise ValueError("Input must be (N,D) or (N,C,H,W)")