import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    # Find most frequent label
    values, counts = np.unique(y_train, return_counts=True)
    majority_label = values[np.argmax(counts)]

    # Number of test samples
    n_test = X_test.shape[0] if X_test.ndim > 1 else len(X_test)

    return np.full(n_test, majority_label)