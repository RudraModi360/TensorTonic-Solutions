import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    cdf=0
    pmf=0
    for i in range(0,k+1):
        temp=comb(n,i)*(p**i)*((1-p)**(n-i))
        cdf+=temp
    pmf=temp
    return float(pmf),float(cdf)
            