import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor, neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns a scalar torch.Tensor: the SGNS loss.
    """
    pos_score=torch.sum(center_vec * pos_vec)
    neg_scores=torch.sum(center_vec * neg_vecs, dim=1)
    loss = -F.logsigmoid(pos_score) - torch.sum(F.logsigmoid(-neg_scores))

    return loss
    