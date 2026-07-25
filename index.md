# cudatensr

`cudatensr` is the dense array foundation of the **cudaverse**: an
experimental GPU-accelerated data science and omics ecosystem designed
around R objects and R package conventions.

## Current capabilities

- Explicit `cpu`, `cuda`, and `auto` device selection.
- Dense vectors, matrices, and arrays through a `cudatensor` S3 class.
- CPU/GPU transfer.
- Natural element-wise arithmetic with safe dtype promotion.
- Matrix multiplication.
- Sum and mean reductions over one-based R dimensions.
- NumPy-style trailing-dimension broadcasting.
- Standard R subsetting and replacement, reshape, transpose, and matrix
  conversion.
- Portable base R backend for development and CI.
- Optional CUDA execution through a CUDA-enabled `torch` installation.

## Installation

``` r

# install.packages("pak")
pak::pak("cudaverse/cudatensr")
```

## Example

``` r

library(cudatensr)

x <- cuda_tensor(matrix(1:6, 2, 3))
y <- cuda_tensor(matrix(1:6, 3, 2))

tensor_device(x)
to_cpu(tensor_matmul(x, y))
to_cpu(x + 0.5)
to_cpu(tensor_mean(x, dim = 1))

reshaped <- tensor_reshape(cuda_tensor(1:12), c(3, 4))
reshaped[1:2, 2:4, drop = FALSE]
t(reshaped)
```

Subsetting always returns a `cudatensor`, including a single selected
value. Replacement preserves the original dtype, so assigning a
fractional value to an integer tensor fails instead of silently
truncating it. In the current release, subsetting a CUDA tensor uses a
CPU round trip; matrix arithmetic, broadcasting, reductions, reshape,
and transpose remain device-native.

Use `device = "cuda"` to require GPU execution:

``` r

if (cuda_available()) {
  x_gpu <- cuda_tensor(matrix(rnorm(1e6), 1000), device = "cuda")
}
```

## Honest backend semantics

`device = "auto"` selects CUDA only when the optional `torch` package
reports a usable CUDA backend; otherwise it selects the base R CPU
backend. The object always reports its actual device and backend through
[`tensor_device()`](https://cudaverse.github.io/cudatensr/reference/tensor_device.md).

Printing tensors with more than 100 values shows metadata without
copying a large CUDA allocation back to R. Use `to_cpu(x)` when you
intentionally want the complete base R array, or change the display
threshold with `options(cudatensr.max_print = 500)`.

This initial release is an API and correctness milestone, not yet a
claim of speedups for every workload.

For installation, device verification, memory advice, and common
failures, see the cudaverse [GPU setup and troubleshooting
guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li
