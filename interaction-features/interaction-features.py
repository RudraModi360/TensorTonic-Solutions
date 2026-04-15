def interaction_features(X):
    """
    Generate pairwise interaction features and append them to the original features.
    """
    results=[]
    for x in X:
        temp=list(x)
        for i in range(len(x)):
            for j in range(i+1,len(x)):
                temp.append(x[i]*x[j])
        results.append(temp)
    return results