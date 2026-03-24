import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # x=np.array(x)
    # p=np.array(p)
    pmf=[]
    for i in range(len(x)):
        if x[i]==1:
             pmf.append(p)
        else:
             pmf.append(1-p)

    return np.array(pmf),p,p*(1-p)