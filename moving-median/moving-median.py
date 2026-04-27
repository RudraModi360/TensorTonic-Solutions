def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    results = []
    
    for i in range(len(values) - window_size + 1):
        temp = sorted(values[i:i + window_size])
        
        if window_size % 2 == 1:
            # odd → middle element
            median = temp[window_size // 2]
        else:
            # even → average of two middle elements
            mid1 = temp[window_size // 2 - 1]
            mid2 = temp[window_size // 2]
            median = (mid1 + mid2) / 2
        
        results.append(median)
    
    return results