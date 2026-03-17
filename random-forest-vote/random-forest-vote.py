import numpy as np

def random_forest_vote(predictions):
    """
    Compute the majority vote from multiple tree predictions.
    
    predictions shape = (n_trees, n_samples)
    """
    
    predictions = np.array(predictions)

    final_preds = []
    
    for i in range(predictions.shape[1]):
        values, counts = np.unique(predictions[:, i], return_counts=True)
        final_preds.append(values[np.argmax(counts)])

    return final_preds