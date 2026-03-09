import math
import numpy as np
def xavier_initialization(W, fan_in, fan_out):
    """
    Scale raw weights to Xavier uniform initialization.
    """
    W=np.array(W)
    L=math.sqrt(6/(fan_in+fan_out))
    return W*2*L-L