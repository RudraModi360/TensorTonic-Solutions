def rating_normalization(matrix):
    """
    Mean-center each user's ratings (ignoring zeros).
    Keeps matrix shape intact.
    """
    results = []

    for row in matrix:
        # extract non-zero ratings
        non_zero = [x for x in row if x != 0]
        
        if len(non_zero) == 0:
            # no ratings → keep row as is
            results.append(row[:])
            continue
        
        mean = sum(non_zero) / len(non_zero)
        
        normalized_row = []
        for x in row:
            if x == 0:
                normalized_row.append(0)
            else:
                normalized_row.append(x - mean)
        
        results.append(normalized_row)

    return results