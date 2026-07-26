# Subset and replace tensor values

Tensor indices follow ordinary one-based R array semantics. Subsetting
returns a `cudatensor`, including when a single value is selected.
Replacement preserves the tensor dtype; fractional values therefore
cannot be assigned to an integer tensor.

## Usage

``` r
# S3 method for class 'cudatensor'
x[..., drop = TRUE]

# S3 method for class 'cudatensor'
x[...] <- value
```

## Arguments

- x:

  A `cudatensor`.

- ...:

  One-based R array indices.

- drop:

  Whether dimensions of length one are dropped.

- value:

  Numeric replacement values or another `cudatensor`.

## Value

A `cudatensor` on the same device as `x`.

## Details

The current implementation performs subsetting and replacement through a
base R array, so a CUDA tensor is transferred to the CPU and the result
is returned to its original device. Use these methods for data
preparation, not inside performance-critical GPU loops.
