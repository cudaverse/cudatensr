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
- Standard R subsetting and replacement, reshape, transpose, and matrix
  conversion.
- Preservation of matrix and array dimension labels across CPU/CUDA transfer,
  arithmetic, compatible broadcasting, reductions, transpose, and matrix
  multiplication.
- Portable base R backend for development and CI.
- Optional CUDA execution through a CUDA-enabled `torch` installation.

## Installation

`cudatensr` is not yet on CRAN. Install the current development release from
GitHub:

```r
# install.packages("pak")
pak::pak("cudaverse/cudatensr")
```

Before a CRAN submission, maintainers run the manual `cran-readiness` workflow
at the exact candidate commit. It checks spelling and URLs, builds one source
tarball with R-devel, runs the full CRAN-style check including the reference
manual, and retains that exact tarball with its check log. CRAN acceptance is
never inferred from an ordinary GitHub check.

## Example

```r
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

Named R matrices keep their identifiers:

```r
named_matrix <- matrix(
  1:6,
  nrow = 2,
  dimnames = list(
    sample = c("sample_a", "sample_b"),
    feature = c("gene_a", "gene_b", "gene_c")
  )
)

tensor <- cuda_tensor(named_matrix)
dimnames(to_cpu(tensor))
```

Element-wise operations require labels on corresponding non-broadcast
dimensions to agree. Matrix multiplication similarly checks labels on the
contracted dimensions, then carries row labels from the left operand and
column labels from the right operand into the result.

Subsetting always returns a `cudatensor`, including a single selected value.
Replacement preserves the original dtype, so assigning a fractional value to
an integer tensor fails instead of silently truncating it. In the current
release, subsetting a CUDA tensor uses a CPU round trip; matrix arithmetic,
broadcasting, reductions, reshape, and transpose remain device-native.

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

Use `cuda_diagnostics()` to inspect the runtime and `cuda_provenance()` to see
what each operation actually did. The
[backend and provenance tutorial](https://cudaverse.github.io/cudatensr/articles/backend-provenance.html)
contains a complete CPU example, an optional CUDA example, transfer and memory
guidance, and the NVIDIA hardware-test contract.

| Request | Actual construction backend | Output device | Fallback |
|---|---|---|---|
| `"cpu"` | CPU / base R | CPU | No |
| `"auto"` with usable CUDA | CUDA / torch | CUDA | No |
| `"auto"` without usable CUDA | CPU / base R | CPU | Yes, recorded with the reason |
| `"cuda"` | CUDA / torch, or a strict error | CUDA when successful | Never silently |

This table describes tensor construction. Later operations can contain several
stages, so the `device` used for computation can differ from the
`output_device` holding the result. The provenance table is authoritative.

Printing tensors with more than 100 values shows metadata without copying a
large CUDA allocation back to R. Use `to_cpu(x)` when you intentionally want
the complete base R array, or change the display threshold with
`options(cudatensr.max_print = 500)`.

This initial release is an API and correctness milestone, not yet a claim of
speedups for every workload.

For installation, device verification, memory advice, and common failures, see
the cudaverse
[GPU setup and troubleshooting guide](https://github.com/cudaverse/.github/blob/main/GPU_SETUP.md).

## License

MIT © Yaoxiang Li
