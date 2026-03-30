import numpy as np

def priority_replay_sample(priorities, alpha, beta):
    """
    Compute sampling probabilities and importance sampling weights for PER.
    """
    priorities=np.array(priorities)
    pow_p=np.pow(priorities,alpha)
    prob_pow_i=pow_p/(np.sum(pow_p))
    w=(prob_pow_i.shape[0]*prob_pow_i)**(-beta)
    return [prob_pow_i.tolist(),(w/np.max(w)).tolist()]