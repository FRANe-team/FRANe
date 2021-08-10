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