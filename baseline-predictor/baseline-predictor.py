def non_zero_sum(matrix):
    answer=0
    count=0
    for mat in matrix:
        for val in mat:
            if val!=0:
                count+=1
                answer+=val
    if count==0:
        return 0.0
    return answer/count
def baseline_predict(ratings_matrix, target_pairs):
    mu = non_zero_sum(ratings_matrix)

    num_users = len(ratings_matrix)
    num_items = len(ratings_matrix[0])

    # --- user bias ---
    user_bias = []
    for u in range(num_users):
        vals = [v for v in ratings_matrix[u] if v != 0]
        if vals:
            user_bias.append(sum(vals)/len(vals) - mu)
        else:
            user_bias.append(0)

    # --- item bias ---
    item_bias = []
    for i in range(num_items):
        vals = [ratings_matrix[u][i] for u in range(num_users) if ratings_matrix[u][i] != 0]
        if vals:
            item_bias.append(sum(vals)/len(vals) - mu)
        else:
            item_bias.append(0)

    # --- predictions ---
    predictions = []
    for u, i in target_pairs:
        predictions.append(mu + user_bias[u] + item_bias[i])

    return predictions