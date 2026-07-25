# cudatensr

`cudatensr` is the dense array foundation of the **cudaverse**: an
experimental GPU-accelerated data science and omics ecosystem designed around
R objects and R package conventions.

## Current capabilities

- Explicit `cpu`, `cuda`, and `auto` device selection.
- Dense vectors, matrices, and arrays through a `cudatensor` S3 class.
- CPU/GPU transfer.
- Natural element-wise arithmetic with safe dtype promotion.
- Matrix multiplication.
- Sum and mean reductions over one-based R dimensions.
- NumPy-style trailing-dimension broadcasting.
- Portable base R backend for development and CI.
- Optional CUDA execution through a CUDA-enabled `torch` installation.

## Installation

```r
# install.packages("pak")
pak::pak("cudaverse/cudatensr")
```

## Example

```r
library(cudatensr)

x <- cuda_tensor(matrix(1:6, 2, 3))
y <- cuda_tensor(matrix(1:6, 3, 2))

tensor_device(x)
to_cpu(tensor_matmul(x, y))
to_cpu(x + 0.5)
to_cpu(tensor_mean(x, dim = 1))
```

Use `device = "cuda"` to require GPU execution:

```r
if (cuda_available()) {
  x_gpu <- cuda_tensor(matrix(rnorm(1e6), 1000), device = "cuda")
}
```

## Honest backend semantics

`device = "auto"` selects CUDA only when the optional `torch` package reports
a usable CUDA backend; otherwise it selects the base R CPU backend. The object
always reports its actual device and backend through `tensor_device()`.

This initial release is an API and correctness milestone, not yet a claim of
speedups for every workload.

## License

MIT © Yaoxiang Li
