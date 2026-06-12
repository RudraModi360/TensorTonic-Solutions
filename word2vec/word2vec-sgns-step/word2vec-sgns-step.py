import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor, center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> tuple:
    """
    Returns tuple (W_in_updated, W_out_updated), each the same shape as the inputs, after one SGNS SGD step.
    """
    v_c=W_in[center_id]
    u_pos=W_out[pos_id]
    u_neg=W_out[neg_ids]

    pos_score=torch.dot(v_c,u_pos)
    neg_score=u_neg@v_c

    pos_coef=torch.sigmoid(pos_score)-1
    neg_coef=torch.sigmoid(neg_score)

    grad_u_pos = pos_coef * v_c
    grad_u_neg = neg_coef.unsqueeze(1) * v_c

    grad_v = (
        pos_coef * u_pos
        +
        torch.sum(
            neg_coef.unsqueeze(1) * u_neg,
            dim=0
        )
    )
    W_in_updated = W_in.clone()
    W_out_updated = W_out.clone()

    W_in_updated[center_id] -= lr * grad_v
    W_out_updated[pos_id] -= lr * grad_u_pos


    for i, nid in enumerate(neg_ids):
        W_out_updated[nid] -= lr * grad_u_neg[i]

    return W_in_updated, W_out_updated