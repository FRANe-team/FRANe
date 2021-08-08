from frane import FRANe
import numpy as np

x = np.random.random((100, 1000))
r = FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)
