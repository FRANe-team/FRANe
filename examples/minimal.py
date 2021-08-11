from frane import FRANe
from scipy.io import loadmat
import numpy as np


# data from https://github.com/jundongl/scikit-feature
x = loadmat('../data/lung_small.mat')["X"].astype(np.float32)
r = FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)
