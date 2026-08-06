# cudaverse

`cudaverse` gives R users one package for GPU-aware numerical workflows. It
combines the former `cudatensr`, `cudasparsr`, `cudalearnr`, `cudagraphR`, and
`cudaembedr` projects behind a single installation and documentation site.

The public API is organized by topic rather than by package:

- device selection and compute provenance;
- dense tensors and sparse matrices;
- SVD, PCA, distances, k-nearest neighbours, and k-means;
- weighted kNN graphs, Louvain, and Leiden clustering;
- UMAP, t-SNE, and diffusion-map-style embeddings.

CUDA is optional. When a supported CUDA-enabled `torch` installation is not
available, functions use their documented portable backend and record what
actually ran.

## Installation

During development, install from GitHub:

```r
# install.packages("pak")
pak::pak("cudaverse/cudaverse")
```

## One workflow, one package

```r
library(cudaverse)

x <- matrix(rnorm(400), nrow = 40)
pca <- cuda_pca(x, n_components = 5)
neighbors <- cuda_knn(pca$x, k = 5)
graph <- cuda_knn_graph(neighbors)
embedding <- cuda_umap(pca$x)

cuda_provenance(pca)
embedding_coordinates(embedding)
```

Single-cell-specific workflows live in the separate `cudacellr` extension so
general users do not need the SingleCellExperiment or Seurat ecosystems.

## Project history

The original component repositories remain available as archived development
history. New features, bug reports, documentation, and releases for the
general-purpose API belong in this repository.
