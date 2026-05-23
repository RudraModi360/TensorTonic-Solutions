import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    positions=np.arange(seq_length)[:,np.newaxis]
    dimensions=np.arange(d_model)[np.newaxis,:]
    denom=1/np.power(10000,(2*(dimensions//2))/d_model)
    angles=positions*denom
    pe = np.zeros((seq_length, d_model))
    pe[:,0::2]=np.sin(angles[:,0::2])
    pe[:,1::2]=np.cos(angles[:,1::2])

    return pe