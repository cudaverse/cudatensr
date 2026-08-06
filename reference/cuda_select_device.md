# Select a computation device without hiding fallback

`"auto"` may select CPU when CUDA is unavailable and records why.
Explicit `"cuda"` is strict: it signals a `cudaverse_cuda_unavailable`
error instead of silently falling back.

## Usage

``` r
cuda_select_device(device = c("auto", "cuda", "cpu"))
```

## Arguments

- device:

  Requested device: `"auto"`, `"cuda"`, or `"cpu"`.

## Value

A named `cuda_device_selection` list containing the original request,
selected device, selection reason, fallback flag, and diagnostics.

## Examples

``` r
cuda_select_device("cpu")
#> <cuda_device_selection requested=cpu selected=cpu reason=explicit_cpu fallback=FALSE>
cuda_select_device("auto")
#> <cuda_device_selection requested=auto selected=cuda reason=cuda_available fallback=FALSE>
```
