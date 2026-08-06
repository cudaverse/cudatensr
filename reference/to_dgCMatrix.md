# Convert to an R sparse matrix

Convert to an R sparse matrix

## Usage

``` r
to_dgCMatrix(x)
```

## Arguments

- x:

  A `cudasparse` matrix.

## Value

A `Matrix::dgCMatrix`.

## Examples

``` r
to_dgCMatrix(cuda_sparse(diag(3), device = "cpu"))
#> 3 x 3 sparse Matrix of class "dgCMatrix"
#>           
#> [1,] 1 . .
#> [2,] . 1 .
#> [3,] . . 1
```
