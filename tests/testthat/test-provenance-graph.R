.graph_test_neighbors <- function(device = "unknown") {
  list(
    index = matrix(
      c(2, 3, 1, 3, 1, 2),
      nrow = 3,
      byrow = TRUE,
      dimnames = list(paste0("cell_", 1:3), NULL)
    ),
    distance = matrix(
      c(1, 2, 1, 1, 2, 1),
      nrow = 3,
      byrow = TRUE,
      dimnames = list(paste0("cell_", 1:3), NULL)
    ),
    device = device
  )
}

test_that("provenance inspection re-exports the canonical generic", {
  expect_identical(cuda_provenance, cuda_provenance)
  expect_true(utils::isS3stdGeneric(cuda_provenance))
})

test_that("graph assembly reports its CPU backend separately from its source", {
  graph <- cuda_knn_graph(
    .graph_test_neighbors("cuda"),
    weighting = "gaussian"
  )
  provenance <- cuda_provenance(graph)

  expect_identical(provenance$stage, "graph_assembly")
  expect_identical(provenance$device, "cpu")
  expect_identical(provenance$backend, "Matrix")
  expect_identical(graph$compute_device, "cpu")
  expect_identical(graph$source_device, "cuda")
  expect_null(graph$source_provenance)
  expect_equal(graph$parameters$sigma, 1)
  expect_output(print(graph), "source_device=cuda compute=cpu")
})

test_that("nearest-neighbour source provenance remains inspectable", {
  x <- matrix(
    c(
      0, 0,
      1, 0,
      0, 1,
      1, 1
    ),
    ncol = 2,
    byrow = TRUE,
    dimnames = list(paste0("cell_", 1:4), NULL)
  )
  neighbors <- cuda_knn(
    x,
    k = 2,
    device = "cpu"
  )
  graph <- cuda_knn_graph(neighbors)

  expect_s3_class(graph$source_provenance, "cuda_provenance")
  expect_identical(
    graph$source_provenance$stage,
    c("distance", "neighbor_selection")
  )
  expect_identical(graph$source_device, "cpu")
  expect_identical(graph$source_class, "cuda_knn")
})

test_that("declared but invalid neighbour provenance is not discarded", {
  neighbors <- .graph_test_neighbors()
  neighbors$compute_stages <- list(
    distance = cuda_stage(
      requested_device = "fixed-cpu",
      device = "cpu",
      backend = "base",
      selection_reason = "algorithm_cpu_only"
    )
  )
  neighbors$compute_device <- "cpu"
  neighbors$provenance_schema <- "cudaverse-stage/99"

  condition <- tryCatch(
    cuda_knn_graph(neighbors),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_provenance_schema_error")

  neighbors$provenance_schema <- "cudaverse-stage/1"
  neighbors$compute_stages <- list(
    broken = list(device = "cpu")
  )
  expect_error(
    cuda_knn_graph(neighbors),
    "does not follow the cudaverse stage schema"
  )
})

test_that("community provenance and effective Leiden parameters are retained", {
  skip_if_not_installed("igraph", minimum_version = "2.0.0")
  graph <- cuda_knn_graph(.graph_test_neighbors())
  louvain <- cuda_louvain(graph)
  leiden <- cuda_leiden(graph, n_iterations = 3)

  expect_identical(
    cuda_provenance(louvain)$stage,
    "community_detection"
  )
  expect_identical(louvain$compute_device, "cpu")
  expect_s3_class(louvain$source_provenance, "cuda_provenance")
  expect_identical(
    louvain$source_provenance$stage,
    "graph_assembly"
  )
  expect_identical(leiden$parameters$n_iterations, 3L)
  expect_output(print(leiden), "compute=cpu backend=igraph")
})

test_that("compatible neighbour device metadata is validated", {
  neighbors <- .graph_test_neighbors()
  neighbors$device <- c("cpu", "cuda")
  expect_error(
    cuda_knn_graph(neighbors),
    "Neighbour `device`"
  )
})
