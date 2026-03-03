import numpy as np

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    classes = np.unique(np.concatenate((y_true, y_pred)))
    K = len(classes)

    label_to_index = {label: i for i, label in enumerate(classes)}
    true_idx = np.array([label_to_index[l] for l in y_true])
    pred_idx = np.array([label_to_index[l] for l in y_pred])

    # Confusion matrix
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        cm[t, p] += 1

    # Accuracy
    accuracy = np.trace(cm) / np.sum(cm)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i in range(K):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(np.sum(cm[i, :]))

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    # Averaging
    if average == "micro":
        TP = np.trace(cm)
        FP = np.sum(cm) - TP
        FN = FP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = precision

    elif average == "macro":
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

    elif average == "weighted":
        weights = supports / np.sum(supports)
        precision = np.sum(precisions * weights)
        recall = np.sum(recalls * weights)
        f1 = np.sum(f1s * weights)

    elif average == "binary":
        if pos_label not in label_to_index:
            raise ValueError("pos_label not found in labels")
        i = label_to_index[pos_label]
        precision = precisions[i]
        recall = recalls[i]
        f1 = f1s[i]

    else:
        raise ValueError("Invalid average type")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }