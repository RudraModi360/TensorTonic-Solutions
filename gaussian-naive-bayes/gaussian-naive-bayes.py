import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    
    classes = np.unique(y_train)
    n_features = X_train.shape[1]
    
    # Store parameters
    means = {}
    variances = {}
    priors = {}
    
    # Compute mean, variance, and prior for each class
    for c in classes:
        X_c = X_train[y_train == c]
        means[c] = X_c.mean(axis=0)
        variances[c] = X_c.var(axis=0) + 1e-9  # avoid division by zero
        priors[c] = X_c.shape[0] / X_train.shape[0]
    
    def gaussian_log_likelihood(x, mean, var):
        return -0.5 * np.sum(np.log(2 * np.pi * var)) - \
               0.5 * np.sum(((x - mean) ** 2) / var)
    
    predictions = []
    
    # Predict each test sample
    for x in X_test:
        posteriors = []
        
        for c in classes:
            prior_log = np.log(priors[c])
            likelihood_log = gaussian_log_likelihood(x, means[c], variances[c])
            posterior = prior_log + likelihood_log
            posteriors.append(posterior)
        
        predictions.append(classes[np.argmax(posteriors)])
    
    return predictions