def double_exponential_smoothing(series, alpha, beta):
    """
    Apply Holt's linear trend method and return the level values.
    """
    n = len(series)
    
    # Initialize
    lev = [series[0]]
    tre = [series[1] - series[0]]
    
    # Result should store level values
    results = [lev[0]]
    
    for i in range(1, n):
        # Compute level
        lt = alpha * series[i] + (1 - alpha) * (lev[i-1] + tre[i-1])
        
        # Compute trend
        bt = beta * (lt - lev[i-1]) + (1 - beta) * tre[i-1]
        
        # Store
        lev.append(lt)
        tre.append(bt)
        results.append(lt)
    
    return results