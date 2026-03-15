import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """

    w = np.array(w)
    grad = np.array(grad)
    E_grad_sq = np.array(E_grad_sq)
    E_update_sq = np.array(E_update_sq)

    # Update running average of gradient square
    E_grad_sq_new = rho * E_grad_sq + (1 - rho) * (grad ** 2)

    # Compute parameter update
    update = - (np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq_new + eps)) * grad

    # Update running average of update square
    E_update_sq_new = rho * E_update_sq + (1 - rho) * (update ** 2)

    # Apply update
    w_new = w + update

    return w_new, E_grad_sq_new, E_update_sq_new