.graph_provenance_metadata <- function(stages) {
  provenance <- cuda_provenance(stages)
  list(
    provenance_schema = attr(provenance, "schema", exact = TRUE),
    compute_device = attr(provenance, "compute_device", exact = TRUE),
    compute_stages = attr(provenance, "compute_stages", exact = TRUE)
  )
}

.with_graph_provenance <- function(x, stages) {
  metadata <- .graph_provenance_metadata(stages)
  x$provenance_schema <- metadata$provenance_schema
  x$compute_device <- metadata$compute_device
  x$compute_stages <- metadata$compute_stages
  x
}

.graph_cpu_stage <- function(backend) {
  cuda_stage(
    requested_device = "fixed-cpu",
    device = "cpu",
    backend = backend,
    selection_reason = "algorithm_cpu_only",
    fallback = FALSE,
    output_device = "cpu"
  )
}

.graph_has_compute_stages <- function(x) {
  if (is.list(x) && "compute_stages" %in% names(x)) {
    return(TRUE)
  }
  !is.null(attr(x, "compute_stages", exact = TRUE))
}

.graph_source_provenance <- function(x) {
  if (!.graph_has_compute_stages(x)) {
    return(NULL)
  }
  cuda_provenance(x)
}

.graph_knn <- function(neighbors) {
  source_class <- class(neighbors)[[1L]]
  source_provenance <- .graph_source_provenance(neighbors)
  if (inherits(neighbors, "cuda_knn")) {
    index <- neighbors$index
    distance <- neighbors$distance
    source_device <- neighbors$device %||% "unknown"
  } else if (is.list(neighbors) &&
             all(c("index", "distance") %in% names(neighbors))) {
    index <- neighbors$index
    distance <- neighbors$distance
    source_device <- neighbors$device %||% "unknown"
  } else {
    stop(
      "`neighbors` must be a cuda_knn object or a list with index and distance.",
      call. = FALSE
    )
  }
  if (!is.character(source_device) || length(source_device) != 1L ||
      is.na(source_device) ||
      !source_device %in% c("cpu", "cuda", "unknown")) {
    stop(
      "Neighbour `device` must be \"cpu\", \"cuda\", or \"unknown\".",
      call. = FALSE
    )
  }
  if (!is.matrix(index) || !is.matrix(distance) ||
      !identical(dim(index), dim(distance)) || nrow(index) < 2L ||
      ncol(index) < 1L) {
    stop("Neighbour index and distance must be same-sized matrices.",
         call. = FALSE)
  }
  if (anyNA(index) || any(index != as.integer(index)) ||
      any(index < 1L) || any(index > nrow(index))) {
    stop("Neighbour indices must identify valid graph vertices.",
         call. = FALSE)
  }
  if (anyNA(distance) || any(!is.finite(distance)) || any(distance < 0)) {
    stop("Neighbour distances must be finite and non-negative.",
         call. = FALSE)
  }
  row_index <- matrix(
    rep(seq_len(nrow(index)), each = ncol(index)),
    nrow = nrow(index),
    byrow = TRUE
  )
  if (any(index == row_index)) {
    stop("Neighbour indices cannot contain self-links.", call. = FALSE)
  }
  duplicate_rows <- vapply(
    seq_len(nrow(index)),
    function(row) anyDuplicated(index[row, ]) > 0L,
    logical(1)
  )
  if (any(duplicate_rows)) {
    stop(
      "Neighbour indices must be unique within each observation; ",
      "duplicate neighbours are not allowed.",
      call. = FALSE
    )
  }
  index_names <- rownames(index)
  distance_names <- rownames(distance)
  if (!is.null(index_names) && !is.null(distance_names) &&
      !identical(index_names, distance_names)) {
    stop(
      "Neighbour index and distance row names must identify the same vertices.",
      call. = FALSE
    )
  }
  vertex_names <- index_names %||% distance_names
  list(
    index = matrix(
      as.integer(index),
      nrow = nrow(index),
      dimnames = dimnames(index)
    ),
    distance = distance,
    device = source_device,
    vertex_names = vertex_names,
    source_class = source_class,
    source_provenance = source_provenance
  )
}

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0L) y else x
}

.graph_weights <- function(distance, weighting, sigma) {
  if (weighting == "binary") {
    return(list(weight = rep(1, length(distance)), sigma = NULL))
  }
  if (weighting == "distance") {
    return(list(weight = 1 / (1 + distance), sigma = NULL))
  }
  if (is.null(sigma)) {
    positive <- distance[distance > 0]
    sigma <- if (length(positive)) stats::median(positive) else 1
  }
  if (!is.numeric(sigma) || length(sigma) != 1L || is.na(sigma) ||
      !is.finite(sigma) || sigma <= 0) {
    stop("`sigma` must be a positive finite number.", call. = FALSE)
  }
  list(
    weight = exp(-(distance^2) / (2 * sigma^2)),
    sigma = sigma
  )
}

#' Build a sparse graph from nearest neighbours
#'
#' The input may have been computed on CUDA, but graph assembly itself is
#' currently performed on the CPU with a sparse `Matrix`.
#' Union graphs retain an edge observed in either direction, whereas mutual
#' graphs require both directed neighbour relations. When the two directions
#' have different weights, the undirected edge retains the stronger affinity.
#'
#' @param neighbors A `cuda_knn()` result or compatible list.
#' @param weighting Edge weighting: binary, inverse-distance, or Gaussian.
#' @param symmetrize Keep the union or only mutual nearest-neighbour edges.
#' @param sigma Gaussian bandwidth. Defaults to the median positive distance.
#' @return A `cuda_graph` list containing sparse `adjacency`, counts of
#'   `vertices` and undirected `edges`, `weighting`, `symmetrize`,
#'   `source_device`, and the graph-assembly `backend`. Named kNN observations
#'   are retained as adjacency dimnames and in `vertex_names`.
#' @export
#' @examples
#' index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
#' distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
#' cuda_knn_graph(list(index = index, distance = distance))
cuda_knn_graph <- function(neighbors,
                           weighting = c("binary", "distance", "gaussian"),
                           symmetrize = c("union", "mutual"),
                           sigma = NULL) {
  neighbors <- .graph_knn(neighbors)
  weighting <- match.arg(weighting)
  symmetrize <- match.arg(symmetrize)
  n <- nrow(neighbors$index)
  k <- ncol(neighbors$index)
  from <- rep(seq_len(n), each = k)
  to <- as.vector(t(neighbors$index))
  distance <- as.vector(t(neighbors$distance))
  weight_result <- .graph_weights(distance, weighting, sigma)
  weight <- weight_result$weight
  sigma <- weight_result$sigma

  low <- pmin(from, to)
  high <- pmax(from, to)
  key <- paste(low, high, sep = ":")
  split_weight <- split(weight, key)
  split_position <- split(seq_along(key), key)
  keep <- if (symmetrize == "mutual") {
    vapply(
      split_position,
      function(position) {
        forward <- from[position] == low[position]
        any(forward) && any(!forward)
      },
      logical(1)
    )
  } else {
    rep(TRUE, length(split_position))
  }
  split_weight <- split_weight[keep]
  split_position <- split_position[keep]
  if (!length(split_weight)) {
    adjacency <- Matrix::sparseMatrix(
      i = integer(), j = integer(), x = numeric(), dims = c(n, n)
    )
  } else {
    edge_weight <- vapply(split_weight, max, numeric(1))
    first <- vapply(split_position, `[`, integer(1), 1L)
    edge_low <- low[first]
    edge_high <- high[first]
    adjacency <- Matrix::sparseMatrix(
      i = c(edge_low, edge_high),
      j = c(edge_high, edge_low),
      x = rep(edge_weight, 2L),
      dims = c(n, n)
    )
  }
  adjacency <- methods::as(adjacency, "dgCMatrix")
  if (!is.null(neighbors$vertex_names)) {
    dimnames(adjacency) <- list(
      neighbors$vertex_names,
      neighbors$vertex_names
    )
  }
  output <- structure(
    list(
      adjacency = adjacency,
      vertices = n,
      edges = length(split_weight),
      weighting = weighting,
      symmetrize = symmetrize,
      source_device = neighbors$device,
      backend = "Matrix",
      vertex_names = neighbors$vertex_names,
      source_class = neighbors$source_class,
      source_provenance = neighbors$source_provenance,
      parameters = list(
        weighting = weighting,
        symmetrize = symmetrize,
        sigma = sigma
      )
    ),
    class = "cuda_graph"
  )
  .with_graph_provenance(
    output,
    list(graph_assembly = .graph_cpu_stage("Matrix"))
  )
}

#' Extract a graph adjacency matrix
#'
#' @param graph A `cuda_graph`.
#' @return A symmetric sparse `Matrix::dgCMatrix`.
#' @export
#' @examples
#' index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
#' distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
#' graph <- cuda_knn_graph(list(index = index, distance = distance))
#' as_adjacency_matrix(graph)
as_adjacency_matrix <- function(graph) {
  if (!inherits(graph, "cuda_graph")) {
    stop("`graph` must be a cuda_graph object.", call. = FALSE)
  }
  graph$adjacency
}

.as_igraph <- function(graph) {
  if (!requireNamespace("igraph", quietly = TRUE)) {
    stop("Install the 'igraph' package to run graph clustering.",
         call. = FALSE)
  }
  if (!inherits(graph, "cuda_graph")) {
    stop("`graph` must be a cuda_graph object.", call. = FALSE)
  }
  igraph::graph_from_adjacency_matrix(
    graph$adjacency,
    mode = "undirected",
    weighted = TRUE,
    diag = FALSE
  )
}

.graph_resolution <- function(resolution) {
  if (!is.numeric(resolution) || length(resolution) != 1L ||
      is.na(resolution) || !is.finite(resolution) || resolution <= 0) {
    stop("`resolution` must be a positive finite number.", call. = FALSE)
  }
  resolution
}

.community_result <- function(fit, graph, igraph_graph, algorithm, resolution,
                              parameters = list()) {
  membership <- as.integer(igraph::membership(fit))
  if (!is.null(graph$vertex_names)) {
    names(membership) <- graph$vertex_names
  }
  output <- structure(
    list(
      membership = membership,
      communities = length(fit),
      modularity = igraph::modularity(
        igraph_graph,
        membership = membership,
        weights = igraph::E(igraph_graph)$weight,
        resolution = resolution
      ),
      algorithm = algorithm,
      resolution = resolution,
      source_device = graph$source_device,
      backend = "igraph",
      source_class = class(graph)[[1L]],
      source_provenance = cuda_provenance(graph),
      upstream_provenance = graph$source_provenance,
      parameters = c(
        list(resolution = resolution),
        parameters
      )
    ),
    class = "cuda_communities"
  )
  .with_graph_provenance(
    output,
    list(community_detection = .graph_cpu_stage("igraph"))
  )
}

#' Cluster a graph with Louvain
#'
#' Community detection currently runs on the CPU through `igraph`.
#'
#' @param graph A `cuda_graph`.
#' @param resolution Positive modularity resolution.
#' @return A `cuda_communities` list containing integer `membership`,
#'   the number of `communities`, `modularity`, `algorithm`, `resolution`,
#'   `source_device`, and clustering `backend`. Membership is named when the
#'   graph has vertex identifiers.
#' @export
#' @examples
#' index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
#' distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
#' graph <- cuda_knn_graph(list(index = index, distance = distance))
#' if (requireNamespace("igraph", quietly = TRUE)) {
#'   cuda_louvain(graph)
#' }
cuda_louvain <- function(graph, resolution = 1) {
  resolution <- .graph_resolution(resolution)
  igraph_graph <- .as_igraph(graph)
  fit <- igraph::cluster_louvain(
    igraph_graph,
    weights = igraph::E(igraph_graph)$weight,
    resolution = resolution
  )
  .community_result(fit, graph, igraph_graph, "louvain", resolution)
}

#' Cluster a graph with Leiden
#'
#' Community detection currently runs on the CPU through `igraph`.
#'
#' @param graph A `cuda_graph`.
#' @param resolution Positive modularity resolution.
#' @param n_iterations Number of Leiden refinement iterations.
#' @return A `cuda_communities` list with the stable fields documented by
#'   [cuda_louvain()].
#' @export
#' @examples
#' index <- matrix(c(2, 3, 1, 3, 1, 2), 3, byrow = TRUE)
#' distance <- matrix(c(1, 2, 1, 1, 2, 1), 3, byrow = TRUE)
#' graph <- cuda_knn_graph(list(index = index, distance = distance))
#' if (requireNamespace("igraph", quietly = TRUE)) {
#'   cuda_leiden(graph)
#' }
cuda_leiden <- function(graph, resolution = 1, n_iterations = 2L) {
  resolution <- .graph_resolution(resolution)
  if (!is.numeric(n_iterations) || length(n_iterations) != 1L ||
      is.na(n_iterations) || n_iterations < 1 ||
      n_iterations != as.integer(n_iterations)) {
    stop("`n_iterations` must be a positive whole number.", call. = FALSE)
  }
  igraph_graph <- .as_igraph(graph)
  fit <- igraph::cluster_leiden(
    igraph_graph,
    objective_function = "modularity",
    weights = igraph::E(igraph_graph)$weight,
    resolution = resolution,
    n_iterations = as.integer(n_iterations)
  )
  .community_result(
    fit,
    graph,
    igraph_graph,
    "leiden",
    resolution,
    parameters = list(n_iterations = as.integer(n_iterations))
  )
}

#' @export
print.cuda_graph <- function(x, ...) {
  cat(sprintf(
    paste0(
      "<cuda_graph vertices=%s edges=%s weighting=%s ",
      "source_device=%s compute=%s backend=%s>\n"
    ),
    x$vertices, x$edges, x$weighting, x$source_device,
    x$compute_device, x$backend
  ))
  invisible(x)
}

#' @export
print.cuda_communities <- function(x, ...) {
  cat(sprintf(
    paste0(
      "<cuda_communities groups=%s algorithm=%s resolution=%s ",
      "compute=%s backend=%s>\n"
    ),
    x$communities, x$algorithm, x$resolution, x$compute_device, x$backend
  ))
  invisible(x)
}
