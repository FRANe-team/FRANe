from frane import FRANe, VERBOSE_LEVEL_NOTHING, VERBOSE_LEVEL_BAR, VERBOSE_LEVEL_VERBOSE
from scipy.io import loadmat
import numpy as np


# data from https://github.com/jundongl/scikit-feature
x = loadmat('../data/lung_small.mat')["X"].astype(np.float32)


def my_metric(x1: np.ndarray, x2: np.ndarray):
    return np.sum(np.square(x1 - x2))


def my_threshold_progression(min_value, max_value, n_steps, _):
    # random thresholds between min and max:
    # we do not need distances (as the fourth argument)
    return np.sort(min_value + np.random.rand(n_steps) * (max_value - min_value))


r = FRANe(
    iterations=200,
    metric=my_metric,
    min_edge_threshold=0.5,
    threshold_function=my_threshold_progression
)
# silent fit
r.fit(x, verbose=VERBOSE_LEVEL_NOTHING)
# progress bar fit
r.fit(x, verbose=VERBOSE_LEVEL_BAR)
# verbose fit
r.fit(x, verbose=VERBOSE_LEVEL_VERBOSE)
scores = r.feature_importances_
print(scores)
