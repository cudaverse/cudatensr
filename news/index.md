# Changelog

## cudaverse 0.1.0

- Establishes one user-facing package for the general-purpose cudaverse
  API.
- Incorporates dense tensor functionality from `cudatensr`.
- Incorporates sparse matrix functionality from `cudasparsr`.
- Incorporates numerical algorithms from `cudalearnr`.
- Incorporates graph workflows from `cudagraphR`.
- Incorporates embedding workflows from `cudaembedr`.
- Preserves the canonical
  [`cuda_provenance()`](https://cudaverse.github.io/cudaverse/reference/cuda_provenance.md)
  protocol across all modules.
- Keeps single-cell-specific workflows in the separate `cudacellr`
  package.
- Fixes CUDA indexing, R column-major reshape semantics, and exact
  self-distance diagonals for compatibility with R torch 0.17.
- Documents the measured, benchmark-gated roadmap toward a lightweight
  native CUDA backend while retaining the current portable CPU fallback.
