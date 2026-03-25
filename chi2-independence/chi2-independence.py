import numpy as np

def chi2_independence(C):
    """
    Compute chi-square test statistic and expected frequencies.
    """
    C = np.array(C)

    row_sum = np.sum(C, axis=1)   # shape (rows,)
    col_sum = np.sum(C, axis=0)   # shape (cols,)
    n = np.sum(C)

    # Outer product to get expected frequencies
    exp = np.outer(row_sum, col_sum) / n

    chi2 = np.sum((C - exp) ** 2 / exp)

    return chi2, exp