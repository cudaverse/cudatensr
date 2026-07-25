# Inspect tensor device and backend

Inspect tensor device and backend

## Usage

``` r
tensor_device(x)
```

## Arguments

- x:

  A `cudatensor`.

## Value

A named character vector.

## Examples

``` r
tensor_device(cuda_tensor(1:3, device = "cpu"))
#>  device backend 
#>   "cpu"  "base" 
```
