import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):

    w = np.array(w)
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)

    m1 = beta1*m + (1-beta1)*grad
    v1 = beta2*v + (1-beta2)*(grad**2)

    w_new = w - lr * (beta1*m1 + (1-beta1)*grad) / (np.sqrt(v1) + eps)

    return w_new, m1, v1