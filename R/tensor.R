#' Detect a usable CUDA backend
#'
#' Detection requires the optional `torch` package and a CUDA-enabled libtorch
#' installation.
#'
#' @return A single logical value.
#' @export
#' @examples
#' cuda_available()
cuda_available <- function() {
  if (!requireNamespace("torch", quietly = TRUE)) {
    return(FALSE)
  }
  isTRUE(tryCatch(torch::cuda_is_available(), error = function(...) FALSE))
}

.tensor_dtype <- function(x) {
  if (is.integer(x)) "integer" else "float64"
}

.validate_integer_values <- function(x, argument = "x") {
  values <- as.numeric(x)
  representable <- values == trunc(values) &
    values >= -.Machine$integer.max &
    values <= .Machine$integer.max
  if (any(!representable)) {
    stop(
      sprintf(
        "`%s` contains values that cannot be represented exactly as integer dtype.",
        argument
      ),
      call. = FALSE
    )
  }
  invisible(x)
}

.new_cudatensor <- function(storage, device, backend, dtype, shape) {
  structure(
    list(
      storage = storage,
      device = device,
      backend = backend,
      dtype = dtype,
      shape = as.integer(shape)
    ),
    class = "cudatensor"
  )
}

.torch_dtype <- function(dtype) {
  switch(
    dtype,
    float32 = torch::torch_float32(),
    float64 = torch::torch_float64(),
    integer = torch::torch_int64()
  )
}

.promote_tensor_dtype <- function(x, y, operation = "arithmetic") {
  ranks <- c(integer = 1L, float32 = 2L, float64 = 3L)
  dtype <- if (ranks[[x]] >= ranks[[y]]) x else y
  if (identical(x, "integer") && identical(y, "integer") &&
      operation %in% c("arithmetic", "division", "power", "matmul")) {
    return("float64")
  }
  dtype
}

.cast_tensor <- function(x, dtype) {
  .check_tensor(x)
  dtype <- match.arg(dtype, c("float64", "float32", "integer"))
  if (identical(x$dtype, dtype)) {
    return(x)
  }
  if (identical(dtype, "integer")) {
    .validate_integer_values(to_cpu(x))
  }
  if (identical(x$backend, "torch")) {
    storage <- x$storage$to(dtype = .torch_dtype(dtype))
    return(.new_cudatensor(
      storage, x$device, x$backend, dtype, x$shape
    ))
  }
  cuda_tensor(to_cpu(x), device = "cpu", dtype = dtype)
}

#' Create a GPU-aware tensor
#'
#' @param x Numeric vector, matrix, array, or another `cudatensor`.
#' @param device One of `"auto"`, `"cuda"`, or `"cpu"`. Auto selects CUDA
#'   only when [cuda_available()] is true.
#' @param dtype One of `"float64"`, `"float32"`, or `"integer"`.
#'
#' @return A `cudatensor` object.
#' @export
#' @examples
#' x <- cuda_tensor(matrix(1:6, nrow = 2), device = "cpu")
#' x
cuda_tensor <- function(x, device = c("auto", "cuda", "cpu"),
                        dtype = NULL) {
  device <- match.arg(device)
  if (inherits(x, "cudatensor")) {
    if (is.null(dtype)) {
      dtype <- x$dtype
    }
    if (device == "auto") {
      device <- if (cuda_available()) "cuda" else "cpu"
    }
    if (identical(device, x$device) && identical(dtype, x$dtype)) {
      return(x)
    }
    x <- to_cpu(x)
  }

  if (!is.numeric(x) || length(x) == 0L || anyNA(x) ||
      any(!is.finite(x))) {
    stop("`x` must contain finite numeric values.", call. = FALSE)
  }
  if (is.null(dim(x))) {
    dim(x) <- length(x)
  }
  if (is.null(dtype)) {
    dtype <- .tensor_dtype(x)
  }
  dtype <- match.arg(dtype, c("float64", "float32", "integer"))
  if (identical(dtype, "integer")) {
    .validate_integer_values(x)
  }
  if (device == "auto") {
    device <- if (cuda_available()) "cuda" else "cpu"
  }
  if (device == "cuda" && !cuda_available()) {
    stop(
      "CUDA is unavailable. Install a CUDA-enabled `torch` backend or use ",
      "`device = \"cpu\"`.",
      call. = FALSE
    )
  }

  shape <- dim(x)
  if (device == "cuda") {
    storage <- torch::torch_tensor(
      x,
      dtype = .torch_dtype(dtype),
      device = "cuda"
    )
    return(.new_cudatensor(storage, "cuda", "torch", dtype, shape))
  }

  storage <- switch(
    dtype,
    integer = array(as.integer(x), dim = shape),
    float32 = array(as.numeric(x), dim = shape),
    float64 = array(as.numeric(x), dim = shape)
  )
  .new_cudatensor(storage, "cpu", "base", dtype, shape)
}

#' Inspect tensor device and backend
#'
#' @param x A `cudatensor`.
#' @return A named character vector.
#' @export
#' @examples
#' tensor_device(cuda_tensor(1:3, device = "cpu"))
tensor_device <- function(x) {
  .check_tensor(x)
  c(device = x$device, backend = x$backend)
}

#' Inspect tensor shape
#'
#' @param x A `cudatensor`.
#' @return An integer vector.
#' @export
#' @examples
#' tensor_shape(cuda_tensor(matrix(1:6, 2), device = "cpu"))
tensor_shape <- function(x) {
  .check_tensor(x)
  x$shape
}

.check_tensor <- function(x, argument = "x") {
  if (!inherits(x, "cudatensor")) {
    stop(sprintf("`%s` must be a `cudatensor`.", argument), call. = FALSE)
  }
  invisible(x)
}

#' Transfer a tensor to a device
#'
#' @param x A `cudatensor`.
#' @param device `"cpu"` or `"cuda"`.
#' @return A `cudatensor` on the requested device.
#' @export
#' @examples
#' x <- cuda_tensor(1:4, device = "cpu")
#' to_device(x, "cpu")
to_device <- function(x, device = c("cpu", "cuda")) {
  .check_tensor(x)
  device <- match.arg(device)
  if (identical(x$device, device)) {
    return(x)
  }
  cuda_tensor(to_cpu(x), device = device, dtype = x$dtype)
}

#' Transfer tensor data to base R
#'
#' @param x A `cudatensor`.
#' @return A base R vector, matrix, or array with the tensor shape.
#' @export
#' @examples
#' to_cpu(cuda_tensor(matrix(1:4, 2), device = "cpu"))
to_cpu <- function(x) {
  .check_tensor(x)
  if (x$backend == "base") {
    result <- x$storage
  } else {
    result <- as.array(x$storage$to(device = "cpu"))
  }
  dim(result) <- x$shape
  result
}

#' Matrix multiplication for tensors
#'
#' @param x,y Two-dimensional `cudatensor` objects or numeric matrices.
#' @return A `cudatensor`.
#' @export
#' @examples
#' x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
#' y <- cuda_tensor(matrix(1:6, 3, 2), device = "cpu")
#' tensor_matmul(x, y)
tensor_matmul <- function(x, y) {
  device <- if (inherits(x, "cudatensor")) {
    x$device
  } else if (inherits(y, "cudatensor")) {
    y$device
  } else {
    "cpu"
  }
  if (!inherits(x, "cudatensor")) {
    x <- cuda_tensor(x, device = device)
  }
  .check_tensor(x)
  if (!inherits(y, "cudatensor")) {
    y <- cuda_tensor(y, device = device)
  }
  .check_tensor(y, "y")
  if (length(x$shape) != 2L || length(y$shape) != 2L) {
    stop("`x` and `y` must both be two-dimensional.", call. = FALSE)
  }
  if (x$shape[[2]] != y$shape[[1]]) {
    stop("Tensor dimensions are not conformable for matrix multiplication.",
         call. = FALSE)
  }
  if (!identical(x$device, y$device)) {
    y <- to_device(y, x$device)
  }
  result_dtype <- .promote_tensor_dtype(x$dtype, y$dtype, "matmul")
  x <- .cast_tensor(x, result_dtype)
  y <- .cast_tensor(y, result_dtype)

  if (x$backend == "torch") {
    storage <- x$storage$matmul(y$storage)
    return(.new_cudatensor(
      storage, x$device, x$backend, result_dtype,
      c(x$shape[[1]], y$shape[[2]])
    ))
  }
  cuda_tensor(
    to_cpu(x) %*% to_cpu(y),
    device = "cpu",
    dtype = result_dtype
  )
}

.tensor_reduce <- function(x, dim, keepdim, fun, torch_method,
                           result_dtype = x$dtype) {
  .check_tensor(x)
  if (!is.logical(keepdim) || length(keepdim) != 1L || is.na(keepdim)) {
    stop("`keepdim` must be TRUE or FALSE.", call. = FALSE)
  }
  if (!is.null(dim)) {
    if (!is.numeric(dim) || anyNA(dim) || any(dim < 1) ||
        any(dim > length(x$shape)) || any(dim != as.integer(dim))) {
      stop("`dim` must contain valid one-based tensor dimensions.",
           call. = FALSE)
    }
    dim <- unique(as.integer(dim))
  }

  if (x$backend == "torch") {
    source_storage <- if (!identical(result_dtype, x$dtype)) {
      x$storage$to(dtype = .torch_dtype(result_dtype))
    } else {
      x$storage
    }
    if (is.null(dim)) {
      storage <- source_storage[[torch_method]]()
    } else {
      storage <- source_storage[[torch_method]](
        dim = dim - 1L,
        keepdim = keepdim
      )
    }
    shape <- as.integer(storage$shape)
    if (length(shape) == 0L) {
      shape <- 1L
    }
    return(.new_cudatensor(
      storage, x$device, x$backend, result_dtype, shape
    ))
  }

  values <- to_cpu(x)
  if (is.null(dim) || length(dim) == length(x$shape)) {
    reduced <- fun(values)
    if (isTRUE(keepdim)) {
      reduced <- array(reduced, dim = rep(1L, length(x$shape)))
    } else {
      reduced <- array(reduced, dim = 1L)
    }
  } else {
    margin <- setdiff(seq_along(x$shape), dim)
    reduced <- apply(values, margin, fun)
    if (isTRUE(keepdim)) {
      kept_shape <- x$shape
      kept_shape[dim] <- 1L
      reduced <- array(reduced, dim = kept_shape)
    }
  }
  cuda_tensor(reduced, device = "cpu", dtype = result_dtype)
}

#' Tensor reductions
#'
#' @param x A `cudatensor`.
#' @param dim Optional one-based dimensions to reduce.
#' @param keepdim Whether reduced dimensions should be retained with size one.
#' @return A `cudatensor`.
#' @export
#' @examples
#' x <- cuda_tensor(matrix(1:6, 2), device = "cpu")
#' tensor_sum(x)
#' tensor_mean(x, dim = 1)
tensor_sum <- function(x, dim = NULL, keepdim = FALSE) {
  result_dtype <- if (identical(x$dtype, "integer")) "float64" else x$dtype
  .tensor_reduce(x, dim, keepdim, sum, "sum", result_dtype)
}

#' @rdname tensor_sum
#' @export
tensor_mean <- function(x, dim = NULL, keepdim = FALSE) {
  result_dtype <- if (identical(x$dtype, "integer")) "float64" else x$dtype
  .tensor_reduce(x, dim, keepdim, mean, "mean", result_dtype)
}

#' Broadcast a tensor to a compatible shape
#'
#' @param x A `cudatensor`.
#' @param shape Target dimensions. Existing dimensions are aligned from the
#'   right and must either match or equal one.
#' @return A `cudatensor`.
#' @export
#' @examples
#' x <- cuda_tensor(1:3, device = "cpu")
#' tensor_broadcast_to(x, c(2, 3))
tensor_broadcast_to <- function(x, shape) {
  .check_tensor(x)
  if (!is.numeric(shape) || length(shape) == 0L || anyNA(shape) ||
      any(shape < 1) || any(shape != as.integer(shape))) {
    stop("`shape` must contain positive whole-number dimensions.",
         call. = FALSE)
  }
  shape <- as.integer(shape)
  if (length(shape) < length(x$shape)) {
    stop("Target shape cannot have fewer dimensions than the tensor.",
         call. = FALSE)
  }
  padded <- c(rep(1L, length(shape) - length(x$shape)), x$shape)
  if (any(padded != 1L & padded != shape)) {
    stop("Tensor shape is not compatible with the target shape.",
         call. = FALSE)
  }

  if (x$backend == "torch") {
    storage <- x$storage$reshape(padded)$expand(shape)
    return(.new_cudatensor(
      storage, x$device, x$backend, x$dtype, shape
    ))
  }

  coordinates <- arrayInd(seq_len(prod(shape)), .dim = shape)
  source_coordinates <- coordinates
  singleton <- padded == 1L
  source_coordinates[, singleton] <- 1L
  strides <- cumprod(c(1L, utils::head(padded, -1L)))
  source_index <- 1L + rowSums(
    sweep(source_coordinates - 1L, 2L, strides, `*`)
  )
  result <- array(as.vector(to_cpu(x))[source_index], dim = shape)
  cuda_tensor(result, device = "cpu", dtype = x$dtype)
}

.broadcast_shape <- function(x, y) {
  size <- max(length(x), length(y))
  x <- c(rep(1L, size - length(x)), x)
  y <- c(rep(1L, size - length(y)), y)
  if (any(x != y & x != 1L & y != 1L)) {
    stop("Tensor shapes are not compatible for broadcasting.", call. = FALSE)
  }
  as.integer(pmax(x, y))
}

.tensor_binary <- function(e1, e2, operator) {
  device <- if (inherits(e1, "cudatensor")) {
    e1$device
  } else if (inherits(e2, "cudatensor")) {
    e2$device
  } else {
    "cpu"
  }
  if (!inherits(e1, "cudatensor")) {
    e1 <- cuda_tensor(e1, device = device)
  }
  if (!inherits(e2, "cudatensor")) {
    e2 <- cuda_tensor(e2, device = device)
  }
  .check_tensor(e1, "e1")
  .check_tensor(e2, "e2")
  if (!identical(e1$device, e2$device)) {
    e2 <- to_device(e2, e1$device)
  }
  operation <- switch(
    operator,
    "/" = "division",
    "^" = "power",
    "arithmetic"
  )
  dtype <- .promote_tensor_dtype(e1$dtype, e2$dtype, operation)
  e1 <- .cast_tensor(e1, dtype)
  e2 <- .cast_tensor(e2, dtype)
  shape <- .broadcast_shape(e1$shape, e2$shape)
  if (!identical(e1$shape, shape)) {
    e1 <- tensor_broadcast_to(e1, shape)
  }
  if (!identical(e2$shape, shape)) {
    e2 <- tensor_broadcast_to(e2, shape)
  }

  if (identical(e1$backend, "torch")) {
    storage <- switch(
      operator,
      "+" = e1$storage + e2$storage,
      "-" = e1$storage - e2$storage,
      "*" = e1$storage * e2$storage,
      "/" = e1$storage / e2$storage,
      "^" = e1$storage^e2$storage
    )
    return(.new_cudatensor(
      storage, e1$device, e1$backend, dtype, shape
    ))
  }

  values <- switch(
    operator,
    "+" = to_cpu(e1) + to_cpu(e2),
    "-" = to_cpu(e1) - to_cpu(e2),
    "*" = to_cpu(e1) * to_cpu(e2),
    "/" = to_cpu(e1) / to_cpu(e2),
    "^" = to_cpu(e1)^to_cpu(e2)
  )
  cuda_tensor(values, device = "cpu", dtype = dtype)
}

#' Arithmetic operators for GPU-aware tensors
#'
#' `cudatensor` objects support element-wise `+`, `-`, `*`, `/`, and `^`.
#' Operands follow trailing-dimension broadcasting. Mixed dtypes are promoted
#' without silently truncating fractional values; integer arithmetic is
#' promoted to `float64` to avoid R integer overflow.
#'
#' Use `%*%` for matrix multiplication.
#'
#' @param e1,e2 A `cudatensor` or finite numeric object for element-wise
#'   arithmetic.
#' @param x,y A `cudatensor` or finite numeric matrix for matrix
#'   multiplication.
#' @return A `cudatensor` on the device of the tensor operand on the left (or
#'   the tensor operand on the right when the left operand is a base object).
#' @name cudatensor-operators
NULL

#' @rdname cudatensor-operators
#' @export
Ops.cudatensor <- function(e1, e2) {
  operator <- .Generic
  supported <- c("+", "-", "*", "/", "^")
  if (!operator %in% supported) {
    stop(
      sprintf(
        "Operator `%s` is not supported for `cudatensor` objects.",
        operator
      ),
      call. = FALSE
    )
  }
  if (missing(e2)) {
    if (identical(operator, "+")) {
      return(e1)
    }
    if (identical(operator, "-")) {
      return(.tensor_binary(0, e1, "-"))
    }
    stop(
      sprintf(
        "Unary operator `%s` is not supported for `cudatensor` objects.",
        operator
      ),
      call. = FALSE
    )
  }
  .tensor_binary(e1, e2, operator)
}

#' @rdname cudatensor-operators
#' @export
`%*%.cudatensor` <- function(x, y) {
  tensor_matmul(x, y)
}

#' @export
dim.cudatensor <- function(x) {
  x$shape
}

#' @export
length.cudatensor <- function(x) {
  prod(x$shape)
}

#' @export
as.array.cudatensor <- function(x, ...) {
  to_cpu(x)
}

#' @export
print.cudatensor <- function(x, ...) {
  cat(
    sprintf(
      "<cudatensor[%s] device=%s backend=%s dtype=%s>\n",
      paste(x$shape, collapse = "x"),
      x$device,
      x$backend,
      x$dtype
    )
  )
  print(to_cpu(x), ...)
  invisible(x)
}
