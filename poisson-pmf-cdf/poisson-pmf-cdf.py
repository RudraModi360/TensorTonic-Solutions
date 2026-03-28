import numpy as np
import math

def poisson_pmf_cdf(lam, k):
    exp_term = np.exp(-lam)

    pmf = (exp_term * lam**k) / math.factorial(k)

    cdf = 0.0
    for i in range(k + 1):
        cdf += (exp_term * lam**i) / math.factorial(i)

    return pmf, cdf