def weighted_moving_average(values, weights):
    """
    Compute the weighted moving average using the given weights.
    """
    window_size = len(weights)
    sum_weights = sum(weights)
    results = []

    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        weighted_sum = sum(w * v for w, v in zip(weights, window))
        results.append(weighted_sum / sum_weights)

    return results