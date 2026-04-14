def target_encoding(categories, targets):
    """
    Replace each category with the mean target value for that category.
    """
    sum_dict = {}
    count_dict = {}

    # Step 1: Compute sums and counts
    for cat, target in zip(categories, targets):
        if cat in sum_dict:
            sum_dict[cat] += target
            count_dict[cat] += 1
        else:
            sum_dict[cat] = target
            count_dict[cat] = 1

    # Step 2: Compute means
    mean_dict = {}
    for cat in sum_dict:
        mean_dict[cat] = sum_dict[cat] / count_dict[cat]

    # Step 3: Encode categories
    encoded = []
    for cat in categories:
        encoded.append(mean_dict[cat])

    return encoded