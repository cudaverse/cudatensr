# Inspect tensor shape

Inspect tensor shape

## Usage

``` r
tensor_shape(x)
```

## Arguments

- x:

  A `cudatensor`.

## Value

An integer vector.

## Examples

``` r
tensor_shape(cuda_tensor(matrix(1:6, 2), device = "cpu"))
#> [1] 2 3
```
