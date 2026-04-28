def seasonal_average(series, period):
    """
    Compute the average value for each position in the seasonal cycle.
    """
    n = len(series)
    m=period
    if n % m != 0:
        raise ValueError("Length of series must be multiple of season length m")

    cycles = n // m
    result = []

    for s in range(m):
        vals = []
        for j in range(cycles):
            vals.append(series[s + j * m])
        result.append(sum(vals) / len(vals))

    return result