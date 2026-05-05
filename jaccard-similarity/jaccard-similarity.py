def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    a=set(set_a)
    b=set(set_b)

    denom=a.union(b)
    num=a.intersection(b)

    if len(denom)==0:
        return 0.0
    return len(num)/len(denom)