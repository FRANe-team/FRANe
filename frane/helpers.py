import scipy.io
import numpy as np
from scipy.spatial.distance import euclidean, canberra, cityblock, minkowski, \
    seuclidean, sqeuclidean, correlation, chebyshev


def normalize(x, metric):
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
    norm = metric(x, 0)
    if 0.0 < norm < float('inf'):
        return x / norm
    else:
        return x

convert_metrices = {
    'euclidean': euclidean,
    'canberra': canberra,
    'cityblock': cityblock,
    'minkowski': minkowski,
    'seuclidean': seuclidean,
    'sqeuclidean': sqeuclidean,
    'correlation': correlation,
    "chebyshev": chebyshev
}
