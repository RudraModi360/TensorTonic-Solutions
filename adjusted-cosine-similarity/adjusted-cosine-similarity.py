from math import sqrt

def find_mean(mat):
    sum_val = 0
    count_val = 0

    for val in mat:
        if val != 0:
            sum_val += val
            count_val += 1

    return sum_val / count_val


def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """

    num = 0
    denom_p1 = 0
    denom_p2 = 0

    for rate in ratings_matrix:

        # User must rate both items
        if rate[item_i] != 0 and rate[item_j] != 0:

            mean = find_mean(rate)

            adj_i = rate[item_i] - mean
            adj_j = rate[item_j] - mean

            num += adj_i * adj_j
            denom_p1 += adj_i ** 2
            denom_p2 += adj_j ** 2

    if denom_p1 == 0 or denom_p2 == 0:
        return 0

    return num / (sqrt(denom_p1) * sqrt(denom_p2))