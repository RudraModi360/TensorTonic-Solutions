def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    X: 2D input matrix (list of lists)
    pool_size: size of pooling window
    stride: stride of the pooling
    """
    
    W_in, H_in = len(X), len(X[0])
    
    W_out = (W_in - pool_size) // stride + 1
    H_out = (H_in - pool_size) // stride + 1
    
    output = [[0 for _ in range(H_out)] for _ in range(W_out)]
    
    for i in range(W_out):
        for j in range(H_out):
            
            w_start = i * stride
            h_start = j * stride
            
            window = []
            
            for m in range(pool_size):
                for n in range(pool_size):
                    window.append(X[w_start + m][h_start + n])
            
            output[i][j] = max(window)
    
    return output