def item_cf_predict(user_ratings, item_similarities, target):
    """
    Predict the rating using item-based collaborative filtering.
    """
    weighted_sum = 0.0
    sim_sum = 0.0

    for i, (rating, sim) in enumerate(zip(user_ratings, item_similarities)):
        if i != target and rating > 0 and sim > 0:
            weighted_sum += rating * sim
            sim_sum += sim

    return weighted_sum / sim_sum if sim_sum > 0 else 0.0