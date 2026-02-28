import numpy as np

def roc_curve(y_true, y_scores):
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Sort by descending score
    desc_sort = np.argsort(-y_scores)
    y_true = y_true[desc_sort]
    y_scores = y_scores[desc_sort]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    tp = 0
    fp = 0

    tpr = [0.0]
    fpr = [0.0]
    thresholds = [np.inf]

    i = 0
    while i < len(y_scores):
        current_score = y_scores[i]

        # Process all samples with same score
        while i < len(y_scores) and y_scores[i] == current_score:
            if y_true[i] == 1:
                tp += 1
            else:
                fp += 1
            i += 1

        # Append once per unique threshold
        tpr.append(tp / P)
        fpr.append(fp / N)
        thresholds.append(current_score)

    return np.array(fpr), np.array(tpr), np.array(thresholds)