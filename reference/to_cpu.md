# Transfer tensor data to base R

Transfer tensor data to base R

## Usage

``` r
to_cpu(x)
```

## Arguments

- x:

  A `cudatensor`.

## Value

A base R vector, matrix, or array with the tensor shape.

## Examples

``` r
to_cpu(cuda_tensor(matrix(1:4, 2), device = "cpu"))
#>      [,1] [,2]
#> [1,]    1    3
#> [2,]    2    4
```
