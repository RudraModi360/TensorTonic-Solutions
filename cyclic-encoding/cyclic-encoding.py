import math

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    if period == 0:
        raise ValueError("period must be non-zero")

    result = []
    for val in values:
        theta = (2 * math.pi * val) / period
        result.append([math.sin(theta), math.cos(theta)])
    
    return result