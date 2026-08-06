# Diagnose the optional CUDA runtime

Inspecting the runtime is non-destructive and never installs or
downloads torch. The returned `reason` is suitable for logs and
provenance.

## Usage

``` r
cuda_diagnostics()
```

## Value

A named list containing `torch_installed`, `torch_version`,
`cuda_available`, `cuda_device_count`, `reason`, and `detection_error`.

## Examples

``` r
cuda_diagnostics()
#> <cuda_diagnostics available=FALSE devices=0 torch=0.17.0 reason=backend_error>
```
