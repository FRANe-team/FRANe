# FRANe

Unsupervised Feature Ranking via Attribute Networks (FRANe)
converts a dataset into a network (graph) with

- nodes that correspond to the features in the data,
- undirected edges whose weights are proportional to the similarity
  between the two corresponding features.

PageRank algorithm is then used to compute the centrality of the nodes
(features) and the computed scores are interpreted as feature importance
scores.
# Instalation
Frane is avalible on pip via `pip install frane`
# Examplary Code Snippet
The FRANe method is implemented in Python3.
The implementation requires some standard scientific libraries
(e.g., numpy and scipy) that make the implementation efficient.

The method is easy to use:

```
import frane
import numpy as np

x = np.random.random((100,1000))
r = frane.FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)
```

See `examples` for more examples. To run tests, please try `pytest ./tests/*`

# Data
The data in the directory `data` was taken from [sk-feature](https://github.com/jundongl/scikit-feature) repository.
