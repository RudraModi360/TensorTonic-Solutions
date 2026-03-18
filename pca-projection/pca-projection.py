import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    X=np.array(X)
    X_mean=np.mean(X,axis=0)
    X_centered=X-X_mean
    n = X.shape[0]
    cov=(X_centered.T@X_centered)/(n-1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx=np.argsort(eigenvalues)[::-1]
    eigenvectors=eigenvectors[:,idx]

    W=eigenvectors[:,:k]
    X_proj = X_centered @ W
    
    return X_proj

    
    