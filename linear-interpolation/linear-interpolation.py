def find_prev(values, i):
    while i >= 0:
        if values[i] is not None:
            return i
        i -= 1
    return None


def find_next(values, i):
    while i < len(values):
        if values[i] is not None:
            return i
        i += 1
    return None


def linear_interpolation(values):
    """
    Fill missing (None) values using linear interpolation.
    """
    values = values[:]  # avoid modifying original list

    for i, val in enumerate(values):
        if val is None:
            left = find_prev(values, i - 1)
            right = find_next(values, i + 1)

            if left is not None and right is not None:
                # Linear interpolation formula
                values[i] = values[left] + (
                    (i - left) / (right - left)
                ) * (values[right] - values[left])

            elif left is not None:
                # Forward fill
                values[i] = values[left]

            elif right is not None:
                # Backward fill
                values[i] = values[right]

    return values