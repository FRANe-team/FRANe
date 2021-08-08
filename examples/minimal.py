import frane
import numpy as np

#from frane import FRANe
#from helpers import load_data

x = np.random.random((100,1000))
r = frane.FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)
