import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    x=np.array(x)
    m=x.shape[0]
    mean=np.mean(x)
    squared_sum=np.sum((x-mean)**2)/(m-1)
    return squared_sum,np.sqrt(squared_sum)