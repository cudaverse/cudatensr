# Tensor reductions

Tensor reductions

## Usage

``` r
tensor_sum(x, dim = NULL, keepdim = FALSE)

tensor_mean(x, dim = NULL, keepdim = FALSE)
```

## Arguments

- x:

  A `cudatensor`.

- dim:

  Optional one-based dimensions to reduce.

- keepdim:

  Whether reduced dimensions should be retained with size one.

## Value

A `cudatensor`.

## Details

Labels on dimensions that are not reduced are retained. A reduced
dimension kept with size one retains its axis name but not its
individual labels. Supplying `integer(0)` performs no reduction and
returns the tensor values, shape, device, and dimnames unchanged (with
the documented reduction dtype promotion).

## Examples

``` r
x <- cuda_tensor(matrix(1:6, 2), device = "cpu")
tensor_sum(x)
#> <cudatensor[1] device=cpu backend=base dtype=float64>
#> [1] 21
tensor_mean(x, dim = 1)
#> <cudatensor[3] device=cpu backend=base dtype=float64>
#> [1] 1.5 3.5 5.5
```
