def polynomial_features(values, degree):
    """
    Generate polynomial features for each value up to the given degree.
    """
    results=[]
    for val in values:
        temp=[]
        for i in range(degree+1):
            temp.append(val**i)
        results.append(temp)
    return results