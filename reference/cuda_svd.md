# GPU-aware singular value decomposition

GPU-aware singular value decomposition

## Usage

``` r
cuda_svd(
  x,
  nu = min(nrow(x), ncol(x)),
  nv = min(nrow(x), ncol(x)),
  device = c("auto", "cuda", "cpu")
)
```

## Arguments

- x:

  A finite numeric matrix or `cudatensor`.

- nu, nv:

  Number of left and right singular vectors to return.

- device:

  One of `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A list with `d`, `u`, `v`, and the actual `device`. Matrix row and
column names are retained on the corresponding singular vectors.

## Examples

``` r
cuda_svd(matrix(rnorm(30), 10, 3), device = "cpu")
#> <cuda_svd rank=3 device=cpu compute=cpu backend=base>
```
