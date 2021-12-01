from frane import FRANe
from scipy.io import loadmat
import numpy as np


# data from https://github.com/jundongl/scikit-feature
x = loadmat('../data/lung_small.mat')["X"].astype(np.float32)
r = FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)

import networkx as nx
import matplotlib.pyplot as plt

plt.figure(1, figsize=(10, 10), dpi=300)
G = nx.convert_matrix.from_numpy_array(r.adjacency)
pos = nx.spring_layout(G)
node_colors = scores
nx.draw_networkx_nodes(G,
                       pos,
                       node_size=10,
                       node_color=node_colors,
                       alpha=0.8)
nx.draw_networkx_edges(G, pos, width=1)
nx.draw_networkx_labels(G, pos, font_size=5, font_color="red")
plt.tight_layout()
plt.show()

