def popularity_ranking(items, min_votes, global_mean):
    """
    Compute the Bayesian weighted rating for each item.
    """
    results = []

    for item in items:
        avg_rating = item[0]
        num_votes = item[1]

        score = (
            (num_votes / (num_votes + min_votes)) * avg_rating
            + (min_votes / (num_votes + min_votes)) * global_mean
        )

        results.append(score)

    return results