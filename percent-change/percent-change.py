def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    results=[]
    for i in range(1,len(series)):
        if series[i-1]!=0:
            results.append((series[i]-series[i-1])/series[i-1])
        else:
            results.append(0.0)
    return results