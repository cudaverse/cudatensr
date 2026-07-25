# Inspect tensor dimension labels

Inspect tensor dimension labels

## Usage

``` r
# S3 method for class 'cudatensor'
dimnames(x)
```

## Arguments

- x:

  A `cudatensor`.

## Value

`NULL` for an unnamed tensor, otherwise one character vector (or `NULL`)
per tensor dimension, following base R
[`dimnames()`](https://rdrr.io/r/base/dimnames.html) semantics.
