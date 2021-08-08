import numpy as np
# TODO to je grdo grdo grdop


def constant_number(start, stop, num, distances, sorted_indexes):
    """
    Returns thresholds, where number of distances between every threshold is the same.

    Returns
    -------
    tresholds : list
    """
    shape = distances.shape[0]  # number of features
    # add the same number of distances every time
    step = (shape**2) // num
    threshold = [
        distances[sorted_indexes[index] // shape,
                  sorted_indexes[index] % shape]
        for index in range(step - 1, shape**2, step)
    ]
    # plus the end
    if threshold[-1] != distances[sorted_indexes[-1] // shape,
                                  sorted_indexes[-1] % shape]:
        threshold.append(distances[sorted_indexes[-1] // shape,
                                   sorted_indexes[-1] % shape])
    return threshold


threshold_dict = {
    'geomspace':
    lambda start, stop, num, distances, sorted_indexes: np.geomspace(
        start, stop, num),
    'logspace':
    lambda start, stop, num, distances, sorted_indexes: np.logspace(
        start, stop, num),
    'linspace':
    lambda start, stop, num, distances, sorted_indexes: np.linspace(
        start, stop, num),
    'median_linear':
    lambda start, stop, num, distances, sorted_indexes: np.linspace(
        start, np.median(distances, axis=None), num),
    'mean_linear':
    lambda start, stop, num, distances, sorted_indexes: np.linspace(
        start, np.mean(distances, axis=None), num),
    'constant_number':
    constant_number
}
