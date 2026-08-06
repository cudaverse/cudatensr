# Cluster a graph with Louvain

Community detection currently runs on the CPU through `igraph`.

## Usage

``` r
cuda_louvain(graph, resolution = 1)
```

## Arguments

- graph:

  A `cuda_graph`.

- resolution:

  Positive modularity resolution.

## Value

A `cuda_communities` list containing integer `membership`, the number of
`communities`, `modularity`, `algorithm`, `resolution`, `source_device`,
and clustering `backend`. Membership is named when the graph has vertex
identifiers.

## Examples

``` r
index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
graph <- cuda_knn_graph(list(index = index, distance = distance))
if (requireNamespace("igraph", quietly = TRUE)) {
  cuda_louvain(graph)
}
#> <cuda_communities groups=1 algorithm=louvain resolution=1 compute=cpu backend=igraph>
```
