import numpy as np

def bootstrap_mean(x, n_bootstrap=1000, ci=0.95, rng=None):
    """
    Returns:
        boot_means: array of bootstrap means
        lower: lower bound of confidence interval
        upper: upper bound of confidence interval
    """
    x = np.array(x)
    n = len(x)
    
    # Random generator (for reproducibility if provided)
    rng = np.random.default_rng(rng)
    
    # Generate bootstrap samples (vectorized)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    samples = x[indices]
    
    # Compute means for each bootstrap sample
    boot_means = samples.mean(axis=1)
    
    # Confidence interval (percentile method)
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    
    return boot_means, lower, upper