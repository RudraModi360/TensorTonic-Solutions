import numpy as np

def mean_average_precision(y_true_list, y_score_list, k=None):

    ap_per_query = []

    for y_true, y_score in zip(y_true_list, y_score_list):

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        # Sort descending
        order = np.argsort(-y_score)
        y_true = y_true[order]

        total_relevant = np.sum(y_true)

        if total_relevant == 0:
            ap_per_query.append(0.0)
            continue

        if k is not None:
            y_true = y_true[:k]

        relevant_so_far = 0
        precision_sum = 0.0

        for i in range(len(y_true)):
            if y_true[i] == 1:
                relevant_so_far += 1
                precision_sum += relevant_so_far / (i + 1)

        ap = precision_sum / total_relevant
        ap_per_query.append(ap)

    if len(ap_per_query) == 0:
        return 0.0, []

    map_value = float(np.mean(ap_per_query))

    return map_value, ap_per_query