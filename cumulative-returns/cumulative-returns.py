def cumulative_returns(returns):
    results = []
    temp = 1.0
    
    for r in returns:
        temp *= (1 + r)
        results.append(temp - 1)
    
    return results