def recursive_ma(values,n,alpha,results):
    if n==0:
        results[n]=values[n]
        return results
    else:
        results[n]=alpha*values[n]+(1-alpha)*recursive_ma(values,n-1,alpha,results)[n-1]
        return results
def exponential_moving_average(values, alpha):
    """
    Compute the exponential moving average of the given values.
    """
    results=values
    return recursive_ma(values,len(values)-1,alpha,results)