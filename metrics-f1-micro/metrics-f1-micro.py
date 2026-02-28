def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    tp = 0
    total = len(y_true)
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            tp += 1
    
    if total == 0:
        return 0.0
    
    return tp / total