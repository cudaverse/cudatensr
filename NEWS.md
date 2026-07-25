# cudatensr 0.1.1

- Added natural element-wise tensor arithmetic and `%*%` dispatch with
  trailing-dimension broadcasting.
- Added explicit mixed-dtype promotion. Integer arithmetic, reductions, and
  matrix products now promote safely instead of truncating fractions or
  silently overflowing to `NA`.
- Lossy explicit conversion to integer dtype now fails clearly.
