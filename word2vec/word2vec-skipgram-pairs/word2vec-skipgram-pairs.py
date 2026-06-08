import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    pairs=[]
    n=token_ids.shape[0]
    for i in range(n):
        left = max(0, i - window)
        right = min(n, i + window + 1)

        for j in range(left, right):
            if i != j:
                pairs.append([token_ids[i].item(), token_ids[j].item()])
    return torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)
                