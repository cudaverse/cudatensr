# Inspect actual compute provenance

Returns one row per computation stage. The table prevents an `"auto"`
request, a CUDA-aware kernel, or a hybrid pipeline from being mistaken
for end-to-end GPU execution.

## Usage

``` r
cuda_provenance(x)
```

## Arguments

- x:

  A cudaverse result or a named list of `cuda_stage` records.

## Value

A `cuda_provenance` data frame with columns `stage`, `requested_device`,
`device`, `backend`, `selection_reason`, `fallback`, and
`output_device`. Its `schema` and `compute_device` attributes contain
the contract version and aggregate actual compute device.

## Examples

``` r
x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
cuda_provenance(x)
#> <cuda_provenance schema=cudaverse-stage/1 stages=1 compute=cpu>
#>                   stage requested_device device backend selection_reason
#>  tensor_materialization              cpu    cpu    base     explicit_cpu
#>  fallback output_device
#>     FALSE           cpu
```
