def normalize(x, metric, p=1):
    """
    Normalize a vector

    Parameters
    ----------
    x : ndarray
        vector
    metric : function
        pair -> float
    
    """
    # norm - distance to 0
    norm = (x, 0)
    if 0.0 < norm < float('inf'):
        return x / norm
    else:
        return x
