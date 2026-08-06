example_knn <- function() {
  list(
    index = matrix(
      c(2, 3, 1, 3, 4, 1, 3, 2),
      nrow = 4,
      byrow = TRUE
    ),
    distance = matrix(
      c(1, 2, 1, 1.5, 0.5, 2, 0.5, 1.5),
      nrow = 4,
      byrow = TRUE
    ),
    device = "cpu"
  )
}

test_that("kNN graphs are sparse, symmetric, and weighted", {
  graph <- cuda_knn_graph(example_knn(), weighting = "distance")
  adjacency <- as_adjacency_matrix(graph)

  expect_s3_class(graph, "cuda_graph")
  expect_s4_class(adjacency, "dgCMatrix")
  expect_identical(dim(adjacency), c(4L, 4L))
  expect_equal(as.matrix(adjacency), t(as.matrix(adjacency)))
  expect_true(all(adjacency@x > 0 & adjacency@x <= 1))
  expect_identical(graph$source_device, "cpu")
})

test_that("mutual graph keeps reciprocal neighbours", {
  graph <- cuda_knn_graph(example_knn(), symmetrize = "mutual")
  expect_lt(graph$edges, cuda_knn_graph(example_knn())$edges)
  expect_equal(as.matrix(graph$adjacency), t(as.matrix(graph$adjacency)))
})

test_that("mutual graph requires both neighbour directions", {
  neighbors <- list(
    index = matrix(
      c(
        2, 3,
        1, 3,
        4, 1,
        3, 2
      ),
      nrow = 4,
      byrow = TRUE
    ),
    distance = matrix(
      c(
        1, 4,
        3, 2,
        2, 1,
        0.5, 5
      ),
      nrow = 4,
      byrow = TRUE
    )
  )

  graph <- cuda_knn_graph(
    neighbors,
    weighting = "distance",
    symmetrize = "mutual"
  )
  adjacency <- as.matrix(as_adjacency_matrix(graph))

  expect_identical(graph$edges, 3L)
  expect_gt(adjacency[1, 2], 0)
  expect_gt(adjacency[1, 3], 0)
  expect_gt(adjacency[3, 4], 0)
  expect_equal(adjacency[2, 3], 0)
  expect_equal(adjacency[2, 4], 0)
})

test_that("asymmetric reciprocal weights retain the stronger affinity", {
  neighbors <- list(
    index = matrix(c(2, 3, 1, 3, 4, 1, 3, 2), 4, byrow = TRUE),
    distance = matrix(c(1, 4, 3, 2, 2, 1, 0.5, 5), 4, byrow = TRUE)
  )

  graph <- cuda_knn_graph(
    neighbors,
    weighting = "distance",
    symmetrize = "mutual"
  )
  adjacency <- as.matrix(as_adjacency_matrix(graph))

  expect_equal(adjacency[1, 2], 1 / (1 + 1))
  expect_equal(adjacency[1, 3], 1 / (1 + 1))
  expect_equal(adjacency[3, 4], 1 / (1 + 0.5))
  expect_equal(adjacency, t(adjacency))
})

test_that("duplicate neighbours are rejected before symmetrization", {
  neighbors <- example_knn()
  neighbors$index[1, 2] <- neighbors$index[1, 1]

  expect_error(
    cuda_knn_graph(neighbors, symmetrize = "union"),
    "duplicate neighbours"
  )
  expect_error(
    cuda_knn_graph(neighbors, symmetrize = "mutual"),
    "duplicate neighbours"
  )
})

test_that("nearest-neighbour results compose with graph workflows", {
  set.seed(1)
  neighbors <- cuda_knn(
    matrix(rnorm(36), 12, 3),
    k = 3,
    device = "cpu",
    batch_size = 2
  )
  graph <- cuda_knn_graph(neighbors)
  expect_identical(graph$vertices, 12L)
})

test_that("vertex identifiers survive graph construction and clustering", {
  neighbors <- example_knn()
  vertex_names <- paste0("cell_", seq_len(nrow(neighbors$index)))
  dimnames(neighbors$index) <- list(vertex_names, c("neighbor_1", "neighbor_2"))
  dimnames(neighbors$distance) <- dimnames(neighbors$index)

  graph <- cuda_knn_graph(neighbors)
  expect_identical(graph$vertex_names, vertex_names)
  expect_identical(
    dimnames(as_adjacency_matrix(graph)),
    list(vertex_names, vertex_names)
  )

  if (requireNamespace("igraph", quietly = TRUE)) {
    set.seed(1)
    communities <- cuda_louvain(graph)
    expect_identical(names(communities$membership), vertex_names)
  }
})

test_that("conflicting neighbour identifiers are rejected", {
  neighbors <- example_knn()
  rownames(neighbors$index) <- paste0("cell_", seq_len(4))
  rownames(neighbors$distance) <- paste0("other_", seq_len(4))

  expect_error(
    cuda_knn_graph(neighbors),
    "same vertices"
  )
})

test_that("Louvain and Leiden return memberships", {
  skip_if_not_installed("igraph")
  graph <- cuda_knn_graph(example_knn())
  set.seed(1)
  louvain <- cuda_louvain(graph)
  set.seed(1)
  leiden <- cuda_leiden(graph)

  expect_s3_class(louvain, "cuda_communities")
  expect_s3_class(leiden, "cuda_communities")
  expect_length(louvain$membership, 4)
  expect_length(leiden$membership, 4)
})

test_that("invalid graph inputs fail clearly", {
  bad <- example_knn()
  bad$index[1, 1] <- 1
  expect_error(cuda_knn_graph(bad), "self-links")
  expect_error(cuda_knn_graph(example_knn(), sigma = -1), NA)
  expect_error(cuda_knn_graph(
    example_knn(), weighting = "gaussian", sigma = -1
  ), "positive finite")
  expect_error(cuda_louvain(
    cuda_knn_graph(example_knn()), resolution = 0
  ), "positive finite")
})
