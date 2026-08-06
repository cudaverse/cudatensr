# Assign observations with a fitted CUDA-aware k-means model

`predict.cuda_kmeans()` computes Euclidean distances to the fitted
centres and returns either the closest-centre assignment or the complete
distance matrix. Named features may be supplied in any order and are
aligned safely.

## Usage

``` r
# S3 method for class 'cuda_kmeans'
predict(
  object,
  newdata,
  type = c("cluster", "distance"),
  device = c("model", "auto", "cuda", "cpu"),
  ...
)
```

## Arguments

- object:

  A fitted `cuda_kmeans` object.

- newdata:

  A finite numeric matrix or data frame with observations in rows and
  model features in columns. When omitted and `type = "cluster"`, the
  training assignments in `object$cluster` are returned.

- type:

  Return closest-centre `"cluster"` assignments or the
  observation-by-centre `"distance"` matrix.

- device:

  Device used for the distance calculation. `"model"` reuses the fitted
  model's actual distance device; `"auto"`, `"cuda"`, and `"cpu"` follow
  the usual cudaverse device-selection rules.

- ...:

  Must be empty.

## Value

For `type = "cluster"`, an integer vector with observation names and,
for recomputed assignments, stage-level provenance. For
`type = "distance"`, a numeric matrix whose columns identify the fitted
centres. Omitting `newdata` returns validated stored training
assignments unchanged and does not create a prediction stage.

## See also

[`cuda_kmeans()`](https://cudaverse.github.io/cudaverse/reference/cuda_kmeans.md)

## Examples

``` r
train <- as.matrix(iris[1:100, 1:4])
fit <- cuda_kmeans(train, centers = 3, seed = 1, device = "cpu")
predict(fit, as.matrix(iris[101:105, 1:4]), device = "cpu")
#> 101 102 103 104 105 
#>   1   1   1   1   1 
#> attr(,"device")
#> [1] "cpu"
#> attr(,"provenance_schema")
#> [1] "cudaverse-stage/1"
#> attr(,"requested_device")
#> [1] "cpu"
#> attr(,"compute_device")
#> [1] "cpu"
#> attr(,"compute_stages")
#> attr(,"compute_stages")$distance
#> $requested_device
#> [1] "cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "base"
#> 
#> $selection_reason
#> [1] "explicit_cpu"
#> 
#> $fallback
#> [1] FALSE
#> 
#> $output_device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_stage"
#> 
#> attr(,"compute_stages")$assignment
#> $requested_device
#> [1] "fixed-cpu"
#> 
#> $device
#> [1] "cpu"
#> 
#> $backend
#> [1] "base"
#> 
#> $selection_reason
#> [1] "algorithm_cpu_only"
#> 
#> $fallback
#> [1] FALSE
#> 
#> $output_device
#> [1] "cpu"
#> 
#> attr(,"class")
#> [1] "cuda_stage"
#> 
#> attr(,"backend")
#> [1] "base"
#> attr(,"parameters")
#> attr(,"parameters")$type
#> [1] "cluster"
#> 
#> attr(,"parameters")$metric
#> [1] "euclidean"
#> 
#> attr(,"source_device")
#> [1] "cpu"
#> attr(,"source_class")
#> [1] "matrix"
```
