# Transfer a tensor to a device

Transfer a tensor to a device

## Usage

``` r
to_device(x, device = c("cpu", "cuda"))
```

## Arguments

- x:

  A `cudatensor`.

- device:

  `"cpu"` or `"cuda"`.

## Value

A `cudatensor` on the requested device.

## Examples

``` r
x <- cuda_tensor(1:4, device = "cpu")
to_device(x, "cpu")
#> <cudatensor[4] device=cpu backend=base dtype=integer>
#> [1] 1 2 3 4
```
