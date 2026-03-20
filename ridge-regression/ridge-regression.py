import numpy as np
def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    X=np.array(X)
    y=np.array(y)
    n=X.shape[-1]
    return np.linalg.inv(X.T@X+lam*np.identity(n))@X.T@y