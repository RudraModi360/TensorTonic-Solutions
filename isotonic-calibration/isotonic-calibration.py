import numpy as np

def calibrate_isotonic(cal_labels, cal_probs, new_probs):
    cal_labels = np.asarray(cal_labels, dtype=float)
    cal_probs = np.asarray(cal_probs, dtype=float)
    new_probs = np.asarray(new_probs, dtype=float)

    # Sort by probability
    order = np.argsort(cal_probs)
    x = cal_probs[order]
    y = cal_labels[order]

    n = len(y)

    # PAVA
    block_values = []
    block_weights = []
    block_starts = []

    for i in range(n):
        block_values.append(y[i])
        block_weights.append(1.0)
        block_starts.append(i)

        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            w1 = block_weights[-2]
            w2 = block_weights[-1]

            merged_weight = w1 + w2
            merged_value = (block_values[-2]*w1 + block_values[-1]*w2) / merged_weight

            block_values[-2] = merged_value
            block_weights[-2] = merged_weight

            block_values.pop()
            block_weights.pop()
            block_starts.pop()

    fitted = np.zeros(n)
    for val, start, weight in zip(block_values, block_starts, block_weights):
        end = start + int(weight)
        fitted[start:end] = val

    calibrated = np.interp(
        new_probs,
        x,
        fitted,
        left=fitted[0],
        right=fitted[-1]
    )

    return list(np.round(calibrated, 6))