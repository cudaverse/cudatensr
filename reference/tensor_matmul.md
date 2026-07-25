# Matrix multiplication for tensors

Matrix multiplication for tensors

## Usage

``` r
tensor_matmul(x, y)
```

## Arguments

- x, y:

  Two-dimensional `cudatensor` objects or numeric matrices.

## Value

A `cudatensor`.

## Examples

``` r
x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
y <- cuda_tensor(matrix(1:6, 3, 2), device = "cpu")
tensor_matmul(x, y)
#> <cudatensor[2x2] device=cpu backend=base dtype=float64>
#>      [,1] [,2]
#> [1,]   22   49
#> [2,]   28   64
```
