# Sparse matrix by dense matrix multiplication

Sparse matrix by dense matrix multiplication

## Usage

``` r
sparse_matmul_dense(x, y)
```

## Arguments

- x:

  A `cudasparse` matrix.

- y:

  A numeric matrix or `cudatensor`.

## Value

A dense `cudatensor`. CUDA multiplication is transferred back to the CPU
in this first release so the result has portable semantics.

## Examples

``` r
x <- cuda_sparse(diag(3), device = "cpu")
sparse_matmul_dense(x, matrix(1:6, 3, 2))
#> <cudatensor[3x2] device=cpu backend=base dtype=float64>
#>      [,1] [,2]
#> [1,]    1    4
#> [2,]    2    5
#> [3,]    3    6
```
