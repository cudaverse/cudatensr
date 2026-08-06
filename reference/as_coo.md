# Convert sparse storage format

Convert sparse storage format

## Usage

``` r
as_coo(x)

as_csr(x)
```

## Arguments

- x:

  A `cudasparse` matrix.

## Value

A `cudasparse` matrix.

## Examples

``` r
x <- cuda_sparse(diag(3), device = "cpu")
as_coo(x)
#> <cudasparse[3x3] nnz=3 format=coo device=cpu backend=Matrix>
#> 3 x 3 sparse Matrix of class "dgCMatrix"
#>           
#> [1,] 1 . .
#> [2,] . 1 .
#> [3,] . . 1
as_csr(x)
#> <cudasparse[3x3] nnz=3 format=csr device=cpu backend=Matrix>
#> 3 x 3 sparse Matrix of class "dgCMatrix"
#>           
#> [1,] 1 . .
#> [2,] . 1 .
#> [3,] . . 1
```
