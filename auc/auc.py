import numpy as np

def auc(fpr, tpr):
    """
    Compute AUC (Area Under ROC Curve) using trapezoidal rule.
    """
    if len(fpr)==len(tpr) and len(tpr)>=2:
        return np.trapezoid(tpr,fpr)
    else:
        return 0.0