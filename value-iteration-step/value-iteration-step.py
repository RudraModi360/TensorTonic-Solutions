import numpy  as np

def value_iteration_step(V, T, R, gamma):
    V,T,R=np.array(V),np.array(T),np.array(R)
    S, A, _ = T.shape
    V_new = np.zeros(S)

    for s in range(S):
        q_values = []

        for a in range(A):
            expected_value = np.dot(T[s, a], V)  
            q = R[s, a] + gamma * expected_value
            q_values.append(q)

        V_new[s] = max(q_values)

    return V_new.tolist()