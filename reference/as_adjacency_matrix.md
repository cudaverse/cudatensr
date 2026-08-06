# Extract a graph adjacency matrix

Extract a graph adjacency matrix

## Usage

``` r
as_adjacency_matrix(graph)
```

## Arguments

- graph:

  A `cuda_graph`.

## Value

A symmetric sparse `Matrix::dgCMatrix`.

## Examples

``` r
index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
graph <- cuda_knn_graph(list(index = index, distance = distance))
as_adjacency_matrix(graph)
#> 3 x 3 sparse Matrix of class "dgCMatrix"
#>           
#> [1,] . 1 1
#> [2,] 1 . 1
#> [3,] 1 1 .
```
