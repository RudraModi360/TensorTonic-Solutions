import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    w=np.array(w)
    v=np.array(v)
    grad=np.array(grad)

    w_look=w-momentum*v

    v_new=momentum*v+(lr*grad)

    w_new=w-v_new

    return w_new,v_new