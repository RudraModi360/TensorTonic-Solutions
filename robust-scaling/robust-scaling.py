def robust_scaling(values):
    """
    Scale values using median and interquartile range (IQR).
    """
    n = len(values)
    
    # Edge case
    if n == 1:
        return [0.0]
    
    # Step 1: sort values
    sorted_vals = sorted(values)
    
    # Helper to compute median
    def median(arr):
        m = len(arr)
        if m % 2 == 1:
            return arr[m // 2]
        else:
            return (arr[m // 2 - 1] + arr[m // 2]) / 2.0
    
    # Step 2: compute median
    med = median(sorted_vals)
    
    # Step 3: split into halves
    if n % 2 == 0:
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2:]
    else:
        lower = sorted_vals[:n // 2]
        upper = sorted_vals[n // 2 + 1:]
    
    # Step 4: compute Q1 and Q3
    q1 = median(lower)
    q3 = median(upper)
    
    # Step 5: compute IQR
    iqr = q3 - q1
    
    # Step 6: scale
    if iqr == 0:
        return [float(v - med) for v in values]
    
    return [float((v - med) / iqr) for v in values]