import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    k=np.array(k)
    pmf=((1-p)**(k-1))*p
    if p!=0:
        mean=1/p
    else:
        mean=0
    return pmf,mean