from frane import FRANe, load_data
import numpy as np

x = np.random.random((100, 1000))
x#  = load_data('../data/lung_small.mat') # data from https://github.com/jundongl/scikit-feature
r = FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)
