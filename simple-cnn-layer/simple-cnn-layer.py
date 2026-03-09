import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride = 1.

    x : (N, C_in, H_in, W_in)
    W : (C_out, C_in, kH, kW)
    b : (C_out,)
    """

    N, C_in, H_in, W_in = x.shape
    C_out, _, kH, kW = W.shape

    H_out = H_in - kH + 1
    W_out = W_in - kW + 1

    out = np.zeros((N, C_out, H_out, W_out))

    for n in range(N):                 # batch
        for c_out in range(C_out):     # filters
            for i in range(H_out):
                for j in range(W_out):

                    patch = x[n, :, i:i+kH, j:j+kW]
                    out[n, c_out, i, j] = np.sum(patch * W[c_out]) + b[c_out]

    return out