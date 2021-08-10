import numpy as np


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
        start, np.mean(distances, axis=None), num)
}
