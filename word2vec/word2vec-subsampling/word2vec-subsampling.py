import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    N=counts.sum()
    counts_ratio=torch.sqrt(t/(counts/N))
    return torch.clamp(counts_ratio,max=1.0)
