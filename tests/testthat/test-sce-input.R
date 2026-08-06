.embedding_test_sce <- function(include_record = TRUE) {
  counts <- matrix(
    seq_len(8L * 12L),
    nrow = 8L,
    dimnames = list(
      paste0("gene_", seq_len(8L)),
      paste0("cell_", seq_len(12L))
    )
  )
  sce <- SingleCellExperiment::SingleCellExperiment(
    assays = list(counts = counts)
  )
  pca <- matrix(
    sin(seq_len(12L * 3L) / 4),
    nrow = 12L,
    dimnames = list(colnames(sce), paste0("PC", seq_len(3L)))
  )
  cudacell_pca <- matrix(
    cos(seq_len(12L * 3L) / 5),
    nrow = 12L,
    dimnames = list(colnames(sce), paste0("CPC", seq_len(3L)))
  )
  SingleCellExperiment::reducedDim(sce, "PCA") <- pca
  SingleCellExperiment::reducedDim(sce, "CUDACELL_PCA") <- cudacell_pca

  if (include_record) {
    metadata <- S4Vectors::metadata(sce)
    metadata$cudacellr <- list(
      schema = "cudacellr-sce/1",
      reduced_dim = "CUDACELL_PCA",
      provenance_schema = "cudaverse-stage/1",
      compute_device = "hybrid",
      compute_stages = list(
        normalization = cuda_stage(
          requested_device = "fixed-cpu",
          device = "cpu",
          backend = "Matrix",
          selection_reason = "algorithm_cpu_only",
          output_device = "cpu"
        ),
        pca_decomposition = cuda_stage(
          requested_device = "cuda",
          device = "cuda",
          backend = "torch",
          selection_reason = "explicit_cuda",
          output_device = "cpu"
        )
      )
    )
    S4Vectors::metadata(sce) <- metadata
  }
  sce
}

.without_cudacell_record <- function(sce) {
  metadata <- S4Vectors::metadata(sce)
  metadata$cudacellr <- NULL
  S4Vectors::metadata(sce) <- metadata
  sce
}

test_that("SingleCellExperiment metadata selects a reduced dimension", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  sce <- .embedding_test_sce()

  input <- .embedding_input(sce)

  expect_identical(input$reduced_dim, "CUDACELL_PCA")
  expect_equal(
    input$matrix,
    unname(as.matrix(
      SingleCellExperiment::reducedDim(sce, "CUDACELL_PCA")
    ))
  )
  expect_identical(input$row_names, colnames(sce))
  expect_identical(input$source_class, "SingleCellExperiment")
  expect_identical(input$source_device, "cuda")
  expect_identical(input$source_compute_device, "hybrid")
  expect_s3_class(input$source_provenance, "cuda_provenance")
  expect_identical(
    input$source_provenance$stage,
    c("normalization", "pca_decomposition")
  )
})

test_that("explicit and conventional reduced-dimension selection is stable", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  sce <- .embedding_test_sce()

  explicit <- .embedding_input(sce, reduced_dim = "PCA")
  expect_identical(explicit$reduced_dim, "PCA")
  expect_null(explicit$source_provenance)
  expect_identical(explicit$source_device, "unknown")
  expect_identical(explicit$source_compute_device, "unknown")

  without_record <- .without_cudacell_record(sce)
  conventional <- .embedding_input(without_record)
  expect_identical(conventional$reduced_dim, "PCA")
  expect_null(conventional$source_provenance)
  expect_identical(conventional$source_device, "unknown")
  expect_identical(conventional$source_compute_device, "unknown")

  only_one <- without_record
  SingleCellExperiment::reducedDims(only_one) <-
    SingleCellExperiment::reducedDims(only_one)["CUDACELL_PCA"]
  expect_error(
    .embedding_input(only_one),
    "reduced_dim.*required.*CUDACELL_PCA"
  )
  unique_choice <- .embedding_input(
    only_one,
    reduced_dim = "CUDACELL_PCA"
  )
  expect_identical(unique_choice$reduced_dim, "CUDACELL_PCA")
})

test_that("unrelated cudacell metadata cannot leak into explicit inputs", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  sce <- .embedding_test_sce()

  metadata <- S4Vectors::metadata(sce)
  metadata$cudacellr$schema <- "cudacellr-sce/99"
  S4Vectors::metadata(sce) <- metadata

  input <- .embedding_input(sce, reduced_dim = "PCA")
  expect_identical(input$reduced_dim, "PCA")
  expect_null(input$source_provenance)
  expect_identical(input$source_device, "unknown")
  expect_identical(input$source_compute_device, "unknown")

  expect_error(
    .embedding_input(sce, reduced_dim = "CUDACELL_PCA"),
    "cudacellr-sce/1"
  )
})

test_that("ambiguous, absent, and stale reduced dimensions fail clearly", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  sce <- .without_cudacell_record(.embedding_test_sce())
  names(SingleCellExperiment::reducedDims(sce)) <- c("TSNE", "UMAP")

  expect_error(
    .embedding_input(sce),
    "reduced_dim.*required.*TSNE.*UMAP"
  )
  expect_error(
    .embedding_input(sce, reduced_dim = "MISSING"),
    "not present.*TSNE.*UMAP"
  )
  expect_error(
    .embedding_input(sce, reduced_dim = character()),
    "one non-empty character"
  )

  no_reduction <- sce
  SingleCellExperiment::reducedDims(no_reduction) <- NULL
  expect_error(
    .embedding_input(no_reduction),
    "no reduced dimensions"
  )

  stale <- .embedding_test_sce()
  metadata <- S4Vectors::metadata(stale)
  metadata$cudacellr$reduced_dim <- "MISSING"
  S4Vectors::metadata(stale) <- metadata
  expect_error(
    .embedding_input(stale),
    "MISSING.*not present"
  )
})

test_that("invalid SCE metadata and reduced values are not ignored", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  sce <- .embedding_test_sce()

  metadata <- S4Vectors::metadata(sce)
  metadata$cudacellr$schema <- "cudacellr-sce/99"
  S4Vectors::metadata(sce) <- metadata
  expect_error(
    .embedding_input(sce),
    "cudacellr-sce/1"
  )

  metadata$cudacellr$schema <- "cudacellr-sce/1"
  metadata$cudacellr$reduced_dim <- NULL
  S4Vectors::metadata(sce) <- metadata
  expect_error(
    .embedding_input(sce),
    "metadata\\(x\\)\\$cudacellr\\$reduced_dim.*one non-empty"
  )

  metadata$cudacellr$reduced_dim <- "CUDACELL_PCA"
  metadata$cudacellr$provenance_schema <- "cudaverse-stage/99"
  S4Vectors::metadata(sce) <- metadata
  condition <- tryCatch(
    .embedding_input(sce),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_provenance_schema_error")

  invalid <- .without_cudacell_record(.embedding_test_sce())
  SingleCellExperiment::reducedDims(invalid) <- list(
    PCA = matrix("not numeric", nrow = ncol(invalid), ncol = 2L)
  )
  expect_error(
    .embedding_input(invalid),
    "finite numeric matrix"
  )
})

test_that("diffusion maps consume SCE input and retain source lineage", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  sce <- .embedding_test_sce()

  fit <- cuda_diffusion_map(
    sce,
    n_components = 2L,
    device = "cpu"
  )

  expect_s3_class(fit, "cuda_embedding")
  expect_identical(rownames(fit$coordinates), colnames(sce))
  expect_identical(fit$source_class, "SingleCellExperiment")
  expect_identical(fit$source_device, "cuda")
  expect_identical(fit$source_compute_device, "hybrid")
  expect_s3_class(fit$source_provenance, "cuda_provenance")
  expect_identical(fit$parameters$reduced_dim, "CUDACELL_PCA")
  expect_identical(fit$compute_device, "cpu")
})

test_that("all embedding entry points expose the SCE selector", {
  expect_true("reduced_dim" %in% names(formals(cuda_umap)))
  expect_true("reduced_dim" %in% names(formals(cuda_tsne)))
  expect_true("reduced_dim" %in% names(formals(cuda_diffusion_map)))

  ordinary <- cuda_diffusion_map(
    matrix(rnorm(30), 10, 3),
    device = "cpu"
  )
  expect_false("reduced_dim" %in% names(ordinary$parameters))

  expect_error(
    .embedding_input(matrix(rnorm(30), 10, 3), reduced_dim = "PCA"),
    "only supported for SingleCellExperiment"
  )
})

test_that("t-SNE accepts a real SingleCellExperiment input", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("Rtsne")
  sce <- .embedding_test_sce()

  fit <- cuda_tsne(
    sce,
    perplexity = 2,
    seed = 1,
    max_iter = 250,
    reduced_dim = "PCA"
  )

  expect_s3_class(fit, "cuda_embedding")
  expect_identical(rownames(fit$coordinates), colnames(sce))
  expect_identical(fit$parameters$reduced_dim, "PCA")
})

test_that("UMAP accepts a real SingleCellExperiment input", {
  skip_if_not_installed("SingleCellExperiment")
  skip_if_not_installed("S4Vectors")
  skip_if_not_installed("uwot")
  sce <- .embedding_test_sce()

  fit <- cuda_umap(
    sce,
    n_neighbors = 4,
    n_epochs = 20,
    seed = 1,
    reduced_dim = "PCA"
  )

  expect_s3_class(fit, "cuda_embedding")
  expect_identical(rownames(fit$coordinates), colnames(sce))
  expect_identical(fit$parameters$reduced_dim, "PCA")
})
