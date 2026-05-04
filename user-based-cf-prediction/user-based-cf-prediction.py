def user_based_cf_prediction(similarities, ratings):
    """
    Predict a rating using user-based collaborative filtering.
    """
    total_sum_num=0
    total_pos_sim=0
    for sim,rat in zip(similarities,ratings):
        if sim*rat>=0:
            total_sum_num+=sim*rat
            total_pos_sim+=sim
    if total_pos_sim!=0:
        return total_sum_num/total_pos_sim
    else:
        return 0.0