from frane import FRANe, load_data
from scipy.io import loadmat
import numpy as np

#x = np.random.random((100, 1000))

# data from https://github.com/jundongl/scikit-feature
x = loadmat('../data/lung_small.mat')["X"].astype(np.float32)
r = FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)
