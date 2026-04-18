from collections import defaultdict

def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    sorted_vals = sorted(values)
    
    # Step 1: store indices for each value
    positions = defaultdict(list)
    for idx, val in enumerate(sorted_vals):
        positions[val].append(idx+1)
    
    # Step 2: compute average rank (this is your "rolling mean" for duplicates)
    avg_ranks = {}
    for val, idxs in positions.items():
        avg_ranks[val] = sum(idxs) / len(idxs)
    
    # Step 3: map back to original values
    result = [avg_ranks[val] for val in values]
    
    return result