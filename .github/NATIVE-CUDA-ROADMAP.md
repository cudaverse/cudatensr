# Lightweight native CUDA roadmap

## Positioning

The long-term CUDA identity of cudaverse is a lightweight, R-native execution
layer rather than a wrapper around a general deep-learning framework:

> Lightweight native CUDA for R -- no Python, no PyTorch, no Conda.

This sentence is a target for the native-backend release. It must not be used
as a claim about cudaverse 0.1.0, whose optional CUDA backend is currently
implemented with R torch.

## Why build a native backend?

The native backend is intended to provide five concrete benefits:

1. **Smaller installation.** CUDA users should not need the complete LibTorch
   runtime. The R torch CUDA installation measured on the Windows RTX 2000
   development machine occupied 6.86 GB. Native releases will publish their
   own measured installed size and list every downloaded component.
2. **Stable R semantics.** cudaverse will own the conversion between R's
   one-based, column-major arrays and device memory instead of adapting to
   changes in another R tensor API's indexing and reshape behavior.
3. **Device-resident workflows.** PCA, pairwise distances, and top-k neighbour
   selection should share device buffers so intermediate results do not make
   avoidable round trips through host memory.
4. **Controlled objects and provenance.** `cudatensor` and `cudasparse` memory
   layout, allocation, finalization, backend version, and stage-level compute
   provenance will be defined by cudaverse contracts.
5. **Backend independence.** The public API and returned object contracts will
   remain stable when native CUDA, CPU, or a future backend is selected.

## Architecture

The public R API will call an internal backend contract rather than a specific
runtime. The initial choices will be `native`, `torch`, and `cpu`, with `auto`
selecting the best validated backend. During migration, torch remains an
optional compatibility backend and the CPU path remains the portable fallback.

The planned native implementation uses NVIDIA's focused numerical libraries
where they are a good fit:

- cuBLAS for dense matrix multiplication;
- cuSPARSE for sparse operations;
- cuSOLVER for decompositions; and
- small cudaverse-owned kernels for reductions, distance calculation,
  broadcasting, and top-k selection.

## Delivery stages

1. Extract and test the internal backend contract without changing the public
   API.
2. Implement device diagnostics, allocation/finalization, transfer, arithmetic,
   reductions, and matrix multiplication.
3. Implement pairwise distance and device-resident top-k selection, then move
   exact k-nearest-neighbour workflows onto the native path.
4. Add sparse operations and decompositions.
5. Make `native` the automatic CUDA choice only after it passes the same CPU/GPU
   parity contract as the current backend.
6. Retain torch for one compatibility cycle, then reassess whether it still
   belongs in `Suggests`.

## Claims require evidence

Lightweight is a measurable contract, not an adjective. Before the native
backend becomes the default, a release candidate must publish reproducible
evidence for:

- clean-machine download and installed size;
- number and identity of external runtime components;
- cold-start time;
- CPU/native/torch parity across supported operations;
- end-to-end benchmarks that include transfers; and
- Windows and Linux installation and hardware checks.

The benchmark report must distinguish resident GPU kernel timing from complete
R workflow timing. It must also report CPU-only and hybrid stages so users can
see where acceleration actually occurred.
