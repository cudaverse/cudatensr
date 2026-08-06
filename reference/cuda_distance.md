# Pairwise distances with an optional CUDA backend

Pairwise distances with an optional CUDA backend

## Usage

``` r
cuda_distance(
  x,
  y = NULL,
  metric = c("euclidean", "cosine"),
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x, y:

  Numeric matrices with observations in rows. When `y` is `NULL`,
  computes all pairwise distances within `x`.

- metric:

  `"euclidean"` or `"cosine"`.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A dense numeric distance matrix with a `device` attribute. Input
observation names are retained as row and column names when present.

## Details

On CPU, Euclidean distances use a common translation and global scaling
before a vectorized calculation. Pairs at risk of cancellation or
non-finite intermediate results are recomputed from direct observation
differences with a scale-first norm. This avoids cancellation from large
shared offsets and avoids avoidable overflow and underflow for extreme
finite values.

## Examples

``` r
cuda_distance(matrix(1:12, 4, 3), device = "cpu")
#>          [,1]     [,2]     [,3]     [,4]
#> [1,] 0.000000 1.732051 3.464102 5.196152
#> [2,] 1.732051 0.000000 1.732051 3.464102
#> [3,] 3.464102 1.732051 0.000000 1.732051
#> [4,] 5.196152 3.464102 1.732051 0.000000
#> attr(,"device")
#> [1] "cpu"
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"requested_device")
#> [1] "cpu"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$distance
#> $requested_device
#> [1] "cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "base"
#> 
#> $selection_reason
#> [1] "explicit_cpu"
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
#> attr(,"backend")
#> [1] "base"
#> attr(,"parameters")
#> attr(,"parameters")$metric
#> [1] "euclidean"
#> 
#> attr(,"source_device")
#> [1] "cpu"
#> attr(,"source_class")
#> [1] "matrix"
```
