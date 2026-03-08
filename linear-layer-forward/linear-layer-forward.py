def linear_layer_forward(X, W, b):
    """
    Compute the forward pass of a linear (fully connected) layer.
    """
    X=np.array(X)
    W=np.array(W)
    b=np.array(b)
    return (X@W+b).tolist()