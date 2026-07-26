# Record one compute stage

`cuda_stage()` is the shared constructor for cudaverse packages and
extensions. It distinguishes the requested device, actual compute
device, implementation backend, and device holding the returned value.

## Usage

``` r
cuda_stage(
  requested_device,
  device,
  backend,
  selection_reason,
  fallback = FALSE,
  output_device = device
)
```

## Arguments

- requested_device:

  `"auto"`, `"cpu"`, `"cuda"`, `"fixed-cpu"`, or `"inherited"`.

- device:

  Actual compute device, `"cpu"` or `"cuda"`.

- backend:

  Concrete implementation backend.

- selection_reason:

  Stable reason describing device selection.

- fallback:

  Whether an `"auto"` request fell back to CPU.

- output_device:

  Device holding the returned value. Defaults to `device`.

## Value

A validated `cuda_stage` list.

## Examples

``` r
cuda_stage(
  requested_device = "auto",
  device = "cpu",
  backend = "base",
  selection_reason = "cuda_unavailable",
  fallback = TRUE
)
#> $requested_device
#> [1] "auto"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "base"
#> 
#> $selection_reason
#> [1] "cuda_unavailable"
#> 
#> $fallback
#> [1] TRUE
#> 
#> $output_device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_stage"
```
