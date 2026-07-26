# Arithmetic operators for GPU-aware tensors

`cudatensor` objects support element-wise `+`, `-`, `*`, `/`, and `^`.
Operands follow trailing-dimension broadcasting. Mixed dtypes are
promoted without silently truncating fractional values; integer
arithmetic is promoted to `float64` to avoid R integer overflow.
Compatible dimension labels are retained. When both operands label the
same non-broadcast dimension, their labels must be identical.

## Usage

``` r
# S3 method for class 'cudatensor'
Ops(e1, e2)

# S3 method for class 'cudatensor'
x %*% y
```

## Arguments

- e1, e2:

  A `cudatensor` or numeric object for element-wise arithmetic.

- x, y:

  A `cudatensor` or numeric matrix for matrix multiplication.

## Value

A `cudatensor` on the device of the tensor operand on the left (or the
tensor operand on the right when the left operand is a base object).

## Details

Use `%*%` for matrix multiplication.

## Examples

``` r
x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
to_cpu(x + c(0.5, 1, 1.5))
#>      [,1] [,2] [,3]
#> [1,]  1.5    4  6.5
#> [2,]  2.5    5  7.5

y <- cuda_tensor(matrix(1:6, 3, 2), device = "cpu")
to_cpu(x %*% y)
#>      [,1] [,2]
#> [1,]   22   49
#> [2,]   28   64
```
