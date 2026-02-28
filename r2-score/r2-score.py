import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Total Sum of Squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Residual Sum of Squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Constant target case
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1 - (ss_res / ss_tot)