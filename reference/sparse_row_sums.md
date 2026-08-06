# Sparse row and column reductions

Sparse row and column reductions

## Usage

``` r
sparse_row_sums(x)

sparse_col_sums(x)
```

## Arguments

- x:

  A `cudasparse` matrix.

## Value

A numeric vector.

## Examples

``` r
x <- cuda_sparse(matrix(1:6, 2), device = "cpu")
sparse_row_sums(x)
#> [1]  9 12
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$row_reduction
#> $requested_device
#> [1] "fixed-cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $selection_reason
#> [1] "algorithm_cpu_only"
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
sparse_col_sums(x)
#> [1]  3  7 11
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$column_reduction
#> $requested_device
#> [1] "fixed-cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "Matrix"
#> 
#> $selection_reason
#> [1] "algorithm_cpu_only"
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
