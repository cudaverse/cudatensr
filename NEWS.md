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
