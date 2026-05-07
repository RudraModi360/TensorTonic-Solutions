from math import log2

def novelty_score(recommendations, item_counts, n_users):
    """
    Compute the average novelty of a recommendation list.
    """

    novelty = []

    for item in recommendations:
        prob = item_counts[item] / n_users
        novelty.append(-log2(prob))

    return sum(novelty) / len(novelty)