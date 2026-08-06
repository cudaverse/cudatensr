# Cluster a graph with Leiden

Community detection currently runs on the CPU through `igraph`.

## Usage

``` r
cuda_leiden(graph, resolution = 1, n_iterations = 2L)
```

## Arguments

- graph:

  A `cuda_graph`.

- resolution:

  Positive modularity resolution.

- n_iterations:

  Number of Leiden refinement iterations.

## Value

A `cuda_communities` list with the stable fields documented by
[`cuda_louvain()`](https://cudaverse.github.io/cudaverse/reference/cuda_louvain.md).

## Examples

``` r
index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
graph <- cuda_knn_graph(list(index = index, distance = distance))
if (requireNamespace("igraph", quietly = TRUE)) {
  cuda_leiden(graph)
}
#> <cuda_communities groups=1 algorithm=leiden resolution=1 compute=cpu backend=igraph>
```
