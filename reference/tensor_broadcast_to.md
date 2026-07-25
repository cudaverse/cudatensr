# Broadcast a tensor to a compatible shape

Broadcast a tensor to a compatible shape

## Usage

``` r
tensor_broadcast_to(x, shape)
```

## Arguments

- x:

  A `cudatensor`.

- shape:

  Target dimensions. Existing dimensions are aligned from the right and
  must either match or equal one.

## Value

A `cudatensor`.

## Examples

``` r
x <- cuda_tensor(1:3, device = "cpu")
tensor_broadcast_to(x, c(2, 3))
#> <cudatensor[2x3] device=cpu backend=base dtype=integer>
#>      [,1] [,2] [,3]
#> [1,]    1    2    3
#> [2,]    1    2    3
```
