# Build a sparse graph from nearest neighbours

The input may have been computed on CUDA, but graph assembly itself is
currently performed on the CPU with a sparse `Matrix`. Union graphs
retain an edge observed in either direction, whereas mutual graphs
require both directed neighbour relations. When the two directions have
different weights, the undirected edge retains the stronger affinity.

## Usage

``` r
cuda_knn_graph(
  neighbors,
  weighting = c("binary", "distance", "gaussian"),
  symmetrize = c("union", "mutual"),
  sigma = NULL
)
```

## Arguments

- neighbors:

  A
  [`cuda_knn()`](https://cudaverse.github.io/cudaverse/reference/cuda_knn.md)
  result or compatible list.

- weighting:

  Edge weighting: binary, inverse-distance, or Gaussian.

- symmetrize:

  Keep the union or only mutual nearest-neighbour edges.

- sigma:

  Gaussian bandwidth. Defaults to the median positive distance.

## Value

A `cuda_graph` list containing sparse `adjacency`, counts of `vertices`
and undirected `edges`, `weighting`, `symmetrize`, `source_device`, and
the graph-assembly `backend`. Named kNN observations are retained as
adjacency dimnames and in `vertex_names`.

## Examples

``` r
index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
cuda_knn_graph(list(index = index, distance = distance))
#> <cuda_graph vertices=3 edges=3 weighting=binary source_device=unknown compute=cpu backend=Matrix>
```
