def differencing(series, order):
    """
    Apply d-th order differencing to the time series.
    """
    data=series
    for i in range(order):
        result=[]
        for j in range(1,len(data)):
            temp=data[j]-data[j-1]
            result.append(temp)
        data=result
    return result
            