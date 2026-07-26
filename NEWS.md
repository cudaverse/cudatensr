# cudatensr 0.2.0

- Added `cuda_diagnostics()` and strict, classed device selection through
  `cuda_select_device()`. Automatic CPU selection now records why CUDA was not
  selected; an explicit CUDA request never silently falls back.
- Added the shared `cudaverse-stage/1` provenance contract. `cuda_stage()` lets
  extensions record stages, while `cuda_provenance()` reports requested,
  actual, backend, fallback, and output-device metadata consistently.
- Tensor construction and public tensor operations now identify their actual
  compute stage. CUDA subsetting and replacement report their CPU round trip
  as hybrid execution.
- CPU `float32` tensor storage now rounds to IEEE single-precision values
  instead of retaining unquantized doubles under a float32 label. Base R
  kernels still accumulate in double precision before rounding their output;
  CUDA kernels use their native torch dtype.
- Floating tensors consistently preserve IEEE `Inf`, `-Inf`, and `NaN`
  values across construction and derived arithmetic. Integer tensors continue
  to reject values that have no exact integer representation.

# cudatensr 0.1.2

- Matrix and array dimnames are now retained as backend-independent metadata
  through construction, CPU/CUDA transfer, compatible arithmetic and
  broadcasting, reductions, transpose, and matrix multiplication. Conflicting
  labels on paired dimensions fail clearly instead of producing a mislabeled
  result.

# cudatensr 0.1.1

- Added standard R subsetting and replacement methods, `tensor_reshape()`,
  `t()`, and `as.matrix()` so tensors can be manipulated without reaching into
  their internal storage.
- Printing a large tensor now reports its metadata without implicitly
  materializing the entire object on the CPU. The display threshold is
  controlled by `options(cudatensr.max_print = 100)`.
- Added natural element-wise tensor arithmetic and `%*%` dispatch with
  trailing-dimension broadcasting.
- Added explicit mixed-dtype promotion. Integer arithmetic, reductions, and
  matrix products now promote safely instead of truncating fractions or
  silently overflowing to `NA`.
- Lossy explicit conversion to integer dtype now fails clearly.
