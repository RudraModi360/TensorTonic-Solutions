import numpy as np

def winsorize(values, lower_pct, upper_pct):
    values = np.asarray(values)

    low_p = np.percentile(values, lower_pct)
    high_p = np.percentile(values, upper_pct)

    return np.clip(values, low_p, high_p).tolist()