def min_max_scaling(data):
    """
    Scale each column of the data matrix to the [0, 1] range.
    """
    if not data:
        return []

    num_rows = len(data)
    num_cols = len(data[0])

    # Create a copy (avoid modifying original)
    scaled = [row[:] for row in data]

    for j in range(num_cols):
        # Extract column
        col = [data[i][j] for i in range(num_rows)]
        
        min_val = min(col)
        max_val = max(col)

        for i in range(num_rows):
            if max_val != min_val:
                scaled[i][j] = (data[i][j] - min_val) / (max_val - min_val)
            else:
                # If all values same → set to 0 (standard practice)
                scaled[i][j] = 0.0

    return scaled