# Inspect sparse matrix dimension labels

Inspect sparse matrix dimension labels

## Usage

``` r
# S3 method for class 'cudasparse'
dimnames(x)
```

## Arguments

- x:

  A `cudasparse` matrix.

## Value

`NULL` for an unnamed matrix, otherwise its row and column names,
following base R [`dimnames()`](https://rdrr.io/r/base/dimnames.html)
semantics.
