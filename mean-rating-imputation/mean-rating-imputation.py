def replacement(mat):
    """
    Replace zeros in a list with the mean of non-zero values.
    """
    # Filter non-zero values
    non_zero_vals = [v for v in mat if v != 0]
    
    # If all values are zero, return as is
    if not non_zero_vals:
        return mat.copy()
    
    # Compute mean excluding zeros
    mean_val = sum(non_zero_vals) / len(non_zero_vals)
    
    # Replace zeros with mean
    return [mean_val if v == 0 else v for v in mat]


def mean_rating_imputation(ratings_matrix, mode="user"):
    """
    Fill missing ratings (zeros) using mean imputation.

    Parameters:
    - ratings_matrix: list of lists (2D matrix)
    - mode: "user" (row-wise) or "item" (column-wise)

    Returns:
    - New matrix with imputed values
    """
    if not ratings_matrix:
        return []

    num_rows = len(ratings_matrix)
    num_cols = len(ratings_matrix[0])

    result = []

    if mode == "user":
        # Row-wise (user-wise)
        for row in ratings_matrix:
            result.append(replacement(row))

    elif mode == "item":
        # Column-wise (item-wise)

        # Step 1: Extract columns
        cols = []
        for j in range(num_cols):
            col = []
            for i in range(num_rows):
                col.append(ratings_matrix[i][j])
            cols.append(col)

        # Step 2: Apply replacement on each column
        cols = [replacement(col) for col in cols]

        # Step 3: Transpose back to rows
        for i in range(num_rows):
            row = []
            for j in range(num_cols):
                row.append(cols[j][i])
            result.append(row)

    else:
        raise ValueError("Mode must be 'user' or 'item'")

    return result