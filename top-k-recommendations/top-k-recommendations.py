def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # filter unrated items
    candidates = [(scores[i], i) for i in range(len(scores)) if i not in rated_indices]
    
    # sort by score descending
    candidates.sort(key=lambda x: -x[0])
    
    # take top-k and return only indices
    return [idx for (_, idx) in candidates[:k]]