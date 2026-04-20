import math

def confusion_matrix(y_true, y_pred):
    tp = tn = fp = fn = 0

    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    # Convert to sorted list of tuples
    return sorted(metrics.items())


def compute_monitoring_metrics(system_type, y_true, y_pred):
    if system_type == "classification":
        return confusion_matrix(y_true, y_pred)

    elif system_type == "regression":
        n = len(y_true)
        mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n if n else 0.0
        rmse = math.sqrt(sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n) if n else 0.0

        metrics = {
            "mae": mae,
            "rmse": rmse
        }

        return sorted(metrics.items())

    elif system_type == "ranking":
        paired = sorted(zip(y_pred, y_true), reverse=True)
        top_k = paired[:3]

        relevant_in_top_k = sum(label for _, label in top_k)
        total_relevant = sum(y_true)

        precision_at_3 = relevant_in_top_k / 3
        recall_at_3 = relevant_in_top_k / total_relevant if total_relevant else 0.0

        metrics = {
            "precision_at_3": precision_at_3,
            "recall_at_3": recall_at_3
        }

        return sorted(metrics.items())

    else:
        return []