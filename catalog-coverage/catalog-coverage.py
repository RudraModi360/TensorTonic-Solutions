def catalog_coverage(recommendations, n_items):
    """
    Compute the catalog coverage of a recommender system.
    """
    temp=[]
    for rec in recommendations:
        temp+=rec
    return len(set(temp))/n_items