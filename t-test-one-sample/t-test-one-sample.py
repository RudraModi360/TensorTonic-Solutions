import numpy as np

def std_cal(x, x_mean):
    n = x.shape[0]
    return np.sqrt(np.sum((x - x_mean)**2) / (n - 1))  # sample std

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.array(x)
    n = x.shape[0]
    x_bar = np.mean(x)

    s = std_cal(x, x_bar)

    t = (x_bar - mu0) / (s / np.sqrt(n))
    return t