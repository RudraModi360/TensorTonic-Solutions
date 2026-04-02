def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    results=[]
    n=len(rewards)

    for i in range(n):
        temp=0
        for j in range(i,n):
            temp+=(gamma**(j-i))*rewards[j]
        results.append(temp)
    return results