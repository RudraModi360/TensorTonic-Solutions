import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    Performs TD(0) update for state-value function.

    Parameters:
    V       : value function (array-like)
    s       : current state
    r       : reward received
    s_next  : next state
    alpha   : learning rate
    gamma   : discount factor

    Returns:
    V_new   : updated value function
    """
    
    V = np.array(V, dtype=float)  
    
    td_target = r + gamma * V[s_next]
    td_error = td_target - V[s]
    
    V[s] += alpha * td_error
    
    return V