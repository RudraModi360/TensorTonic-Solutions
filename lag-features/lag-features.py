def lag_features(series, lags):
    """
    Create a lag feature matrix from the time series.
    """
    results = []
    max_lag = max(lags)
    
    for i in range(max_lag, len(series)):
        row = []
        for lag in lags:
            row.append(series[i - lag])
        results.append(row)
    
    return results