from math import sqrt

def rolling_std(values, window_size):
    """
    Compute the rolling population standard deviation.
    """
    results = []
    
    for i in range(len(values) - window_size + 1):
        temp = values[i:i + window_size]
        temp_mean = sum(temp) / window_size
        
        temp_std = 0.0
        for j in range(window_size):
            temp_std += (temp[j] - temp_mean) ** 2
        
        variance = temp_std / window_size
        results.append(sqrt(variance))
    
    return results