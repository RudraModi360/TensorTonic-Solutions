def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    if not recommendations or not ground_truth or k <= 0:
        return 0.0
    
    hit = 0
    total = len(ground_truth)

    for rec, truth in zip(recommendations, ground_truth):
        if type(truth)==list:
            truth=truth[0]
        if truth in rec[:k]:
            hit += 1

    return hit / total