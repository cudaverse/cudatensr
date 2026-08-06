# Inspect sparse matrix metadata

Inspect sparse matrix metadata

## Usage

``` r
sparse_info(x)
```

## Arguments

- x:

  A `cudasparse` matrix.

## Value

A named list containing `shape`, `nnz`, `density`, `format`, actual
`device`, and `backend`.

## Examples

``` r
sparse_info(cuda_sparse(diag(3), device = "cpu"))
#> $shape
#> [1] 3 3
#> 
#> $nnz
#> [1] 3
#> 
#> $density
#> [1] 0.3333333
#> 
#> $format
#> [1] "csr"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $provenance_schema
#> [1] "cudaverse-stage/1"
#> 
#> $compute_device
#> [1] "cpu"
#> 
```
