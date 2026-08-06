# Sparse matrix-vector multiplication

Sparse matrix-vector multiplication

## Usage

``` r
sparse_matvec(x, y)
```

## Arguments

- x:

  A `cudasparse` matrix.

- y:

  A numeric vector.

## Value

A numeric vector.

## Examples

``` r
sparse_matvec(cuda_sparse(diag(3), device = "cpu"), 1:3)
#> [1] 1 2 3
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$sparse_multiply
#> $requested_device
#> [1] "inherited"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $selection_reason
#> [1] "inherited_device"
#> 
#> $fallback
#> [1] FALSE
#> 
#> $output_device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_stage"
#> 
```
