# cudaverse

`cudaverse` gives R users one package for GPU-aware numerical workflows.
It combines the former `cudatensr`, `cudasparsr`, `cudalearnr`,
`cudagraphR`, and `cudaembedr` projects behind a single installation and
documentation site.

The public API is organized by topic rather than by package:

- device selection and compute provenance;
- dense tensors and sparse matrices;
- SVD, PCA, distances, k-nearest neighbours, and k-means;
- weighted kNN graphs, Louvain, and Leiden clustering;
- UMAP, t-SNE, and diffusion-map-style embeddings.

CUDA is optional. When a supported CUDA-enabled `torch` installation is
not available, functions use their documented portable backend and
record what actually ran.

## Lightweight native CUDA direction

Version 0.1.0 uses `torch` only as an optional CUDA backend; it is not a
hard package dependency. The next backend milestone is a lightweight
native CUDA implementation behind the same public R API. Its intended
benefits are:

- avoid requiring the full LibTorch installation, which occupied 6.86 GB
  in our Windows RTX 2000 development environment;
- remove coupling to changes in torch’s R indexing, reshape semantics,
  and release cycle;
- keep PCA, distance calculation, and top-k selection resident on the
  GPU instead of repeatedly transferring intermediate results;
- control `cudatensor` and `cudasparse` memory layout, lifetime, and
  compute provenance directly; and
- allow future backends to be added without changing user code.

These are roadmap goals, not performance or installation-size claims
about the current release. They will become release claims only after
reproducible parity, disk-footprint, and performance benchmarks pass.
See the [native CUDA
roadmap](https://cudaverse.github.io/cudaverse/NATIVE-CUDA-ROADMAP.md)
for the architecture and acceptance criteria.

## Installation

During development, install from GitHub:

``` r
# install.packages("pak")
pak::pak("cudaverse/cudaverse")
```

## One workflow, one package

``` r
library(cudaverse)

x <- matrix(rnorm(400), nrow = 40)
pca <- cuda_pca(x, n_components = 5)
neighbors <- cuda_knn(pca$x, k = 5)
graph <- cuda_knn_graph(neighbors)
embedding <- cuda_umap(pca$x)

cuda_provenance(pca)
embedding_coordinates(embedding)
```

Single-cell-specific workflows live in the separate `cudacellr`
extension so general users do not need the SingleCellExperiment or
Seurat ecosystems.

## Project history

The original component repositories remain available as archived
development history. New features, bug reports, documentation, and
releases for the general-purpose API belong in this repository.
