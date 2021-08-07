# FRANe

Unsupervised Feature Ranking via Attribute Networks (FRANe)
converts a dataset into a network (graph) with

- nodes that correspond to the features in the data,
- undirected edges whose weights are proportional to the similarity
  between the two corresponding features.

PageRank algorithm is than used to compute the centrality of the nodes
(features) and the computed scores are interpreted as feature importance
scores.

![Overview of FRANe](https://github.com/FRANe-team/FRANe/blob/main/sketch.png)

# Examplary Code Snippet
The FRANe method is implemented in Python3.
The implementation requires some standard scientific libraries
(e.g., numpy and scipy) that make the implementation efficient.

The method is easy to use:

```
from frane import FRANe
from helpers import load_data


x = load_data("../data/lung_small.mat")
r = FRANe()
r.fit(x)
scores = r.feature_importances_
print(scores)
```

See `code/example.py` for a more detailed example.
