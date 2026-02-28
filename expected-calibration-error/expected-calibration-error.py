import numpy as np

def expected_calibration_error(y_true, y_pred, n_bins=5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]

        # Include right edge in last bin
        if i == n_bins - 1:
            in_bin = (y_pred >= bin_lower) & (y_pred <= bin_upper)
        else:
            in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)

        if np.any(in_bin):
            bin_true = y_true[in_bin]
            bin_pred = y_pred[in_bin]

            # 🔥 Correct academic definition
            bin_accuracy = np.mean(bin_true)
            bin_confidence = np.mean(bin_pred)

            bin_weight = len(bin_true) / n
            ece += np.abs(bin_accuracy - bin_confidence) * bin_weight

    return float(ece)