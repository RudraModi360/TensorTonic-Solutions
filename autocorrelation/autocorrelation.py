def autocorrelation(series, max_lag):
    """
    Compute the autocorrelation of a time series for lags 0 to max_lag.
    """
    curr_mean=0.0
    curr_std=0.0
    for val in series:
        curr_mean+=val

    curr_mean/=len(series)

    for val in series:
        curr_std+=(val-curr_mean)**2

    autocorr=[]

    if curr_std == 0:
        return [1.0] + [0.0] * max_lag

    for k in range(max_lag + 1):
        num = 0.0
        for t in range(len(series) - k):
            num += (series[t] - curr_mean) * (series[t + k] - curr_mean)

        autocorr.append(num / curr_std)


    # autocorr/=curr_std
    return autocorr