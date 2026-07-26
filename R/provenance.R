.cudaverse_provenance_schema <- "cudaverse-stage/1"
.cuda_stage_fields <- c(
  "requested_device",
  "device",
  "backend",
  "selection_reason",
  "fallback",
  "output_device"
)

.scalar_character <- function(x, argument, allow_na = FALSE) {
  valid <- is.character(x) && length(x) == 1L && is.null(dim(x)) &&
    !is.object(x)
  if (valid && allow_na && is.na(x)) {
    return(x)
  }
  if (!valid || is.na(x) || !nzchar(x)) {
    stop(
      sprintf(
        "`%s` must be %sone non-empty character string.",
        argument,
        if (allow_na) "NA or " else ""
      ),
      call. = FALSE
    )
  }
  x
}

.cuda_diagnostic_reason <- function(torch_installed, available,
                                    detection_error) {
  if (!torch_installed) {
    return("torch_not_installed")
  }
  if (!is.null(detection_error)) {
    return("backend_error")
  }
  if (available) "cuda_available" else "cuda_unavailable"
}

.cuda_torch_installed <- function() {
  requireNamespace("torch", quietly = TRUE)
}

.cuda_torch_version <- function() {
  as.character(utils::packageVersion("torch"))
}

.cuda_torch_is_available <- function() {
  torch::cuda_is_available()
}

.cuda_torch_device_count <- function() {
  torch::cuda_device_count()
}

.valid_cuda_device_count <- function(x) {
  is.numeric(x) &&
    length(x) == 1L &&
    !is.na(x) &&
    is.finite(x) &&
    x >= 0 &&
    x <= .Machine$integer.max &&
    x == trunc(x)
}

#' Diagnose the optional CUDA runtime
#'
#' Inspecting the runtime is non-destructive and never installs or downloads
#' torch. The returned `reason` is suitable for logs and provenance.
#'
#' @return A named list containing `torch_installed`, `torch_version`,
#'   `cuda_available`, `cuda_device_count`, `reason`, and `detection_error`.
#' @export
#' @examples
#' cuda_diagnostics()
cuda_diagnostics <- function() {
  torch_installed <- .cuda_torch_installed()
  torch_version <- if (torch_installed) {
    .cuda_torch_version()
  } else {
    NA_character_
  }
  detection_error <- NULL
  available <- if (torch_installed) {
    tryCatch(
      isTRUE(.cuda_torch_is_available()),
      error = function(error) {
        detection_error <<- conditionMessage(error)
        FALSE
      }
    )
  } else {
    FALSE
  }
  device_count <- if (available) {
    raw_device_count <- tryCatch(
      .cuda_torch_device_count(),
      error = function(error) {
        if (is.null(detection_error)) {
          detection_error <<- conditionMessage(error)
        }
        NA_real_
      }
    )
    if (!.valid_cuda_device_count(raw_device_count)) {
      if (is.null(detection_error)) {
        detection_error <- paste0(
          "`torch::cuda_device_count()` returned an invalid value; ",
          "expected one non-negative whole number."
        )
      }
      available <- FALSE
      NA_integer_
    } else {
      as.integer(raw_device_count)
    }
  } else {
    0L
  }
  if (length(device_count) == 1L &&
      !is.na(device_count) &&
      device_count < 1L) {
    available <- FALSE
  }

  structure(
    list(
      torch_installed = torch_installed,
      torch_version = torch_version,
      cuda_available = available,
      cuda_device_count = device_count,
      reason = .cuda_diagnostic_reason(
        torch_installed,
        available,
        detection_error
      ),
      detection_error = detection_error
    ),
    class = "cuda_diagnostics"
  )
}

#' Select a computation device without hiding fallback
#'
#' `"auto"` may select CPU when CUDA is unavailable and records why.
#' Explicit `"cuda"` is strict: it signals a `cudaverse_cuda_unavailable`
#' error instead of silently falling back.
#'
#' @param device Requested device: `"auto"`, `"cuda"`, or `"cpu"`.
#' @return A named `cuda_device_selection` list containing the original
#'   request, selected device, selection reason, fallback flag, and diagnostics.
#' @export
#' @examples
#' cuda_select_device("cpu")
#' cuda_select_device("auto")
cuda_select_device <- function(device = c("auto", "cuda", "cpu")) {
  requested_device <- match.arg(device)
  if (identical(requested_device, "cpu")) {
    return(structure(
      list(
        requested_device = "cpu",
        device = "cpu",
        selection_reason = "explicit_cpu",
        fallback = FALSE,
        diagnostics = NULL
      ),
      class = "cuda_device_selection"
    ))
  }

  diagnostics <- cuda_diagnostics()
  if (isTRUE(diagnostics$cuda_available)) {
    return(structure(
      list(
        requested_device = requested_device,
        device = "cuda",
        selection_reason = if (identical(requested_device, "cuda")) {
          "explicit_cuda"
        } else {
          "cuda_available"
        },
        fallback = FALSE,
        diagnostics = diagnostics
      ),
      class = "cuda_device_selection"
    ))
  }

  if (identical(requested_device, "cuda")) {
    message <- paste0(
      "CUDA is unavailable (", diagnostics$reason, "). ",
      "Install a CUDA-enabled `torch` backend or use `device = \"cpu\"`."
    )
    condition <- structure(
      list(
        message = message,
        call = NULL,
        diagnostics = diagnostics
      ),
      class = c(
        "cudaverse_cuda_unavailable",
        "error",
        "condition"
      )
    )
    stop(condition)
  }

  structure(
    list(
      requested_device = "auto",
      device = "cpu",
      selection_reason = diagnostics$reason,
      fallback = TRUE,
      diagnostics = diagnostics
    ),
    class = "cuda_device_selection"
  )
}

#' Record one compute stage
#'
#' `cuda_stage()` is the shared constructor for cudaverse packages and
#' extensions. It distinguishes the requested device, actual compute device,
#' implementation backend, and device holding the returned value.
#'
#' @param requested_device `"auto"`, `"cpu"`, `"cuda"`, `"fixed-cpu"`, or
#'   `"inherited"`.
#' @param device Actual compute device, `"cpu"` or `"cuda"`.
#' @param backend Concrete implementation backend.
#' @param selection_reason Stable reason describing device selection.
#' @param fallback Whether an `"auto"` request fell back to CPU.
#' @param output_device Device holding the returned value. Defaults to `device`.
#' @return A validated `cuda_stage` list.
#' @export
#' @examples
#' cuda_stage(
#'   requested_device = "auto",
#'   device = "cpu",
#'   backend = "base",
#'   selection_reason = "cuda_unavailable",
#'   fallback = TRUE
#' )
cuda_stage <- function(requested_device, device, backend, selection_reason,
                       fallback = FALSE, output_device = device) {
  requested_device <- .scalar_character(
    requested_device,
    "requested_device"
  )
  device <- .scalar_character(device, "device")
  backend <- .scalar_character(backend, "backend")
  selection_reason <- .scalar_character(
    selection_reason,
    "selection_reason"
  )
  output_device <- .scalar_character(output_device, "output_device")
  if (!requested_device %in%
      c("auto", "cpu", "cuda", "fixed-cpu", "inherited")) {
    stop(
      paste0(
        "`requested_device` must be \"auto\", \"cpu\", \"cuda\", ",
        "\"fixed-cpu\", or \"inherited\"."
      ),
      call. = FALSE
    )
  }
  if (!device %in% c("cpu", "cuda")) {
    stop("`device` must be \"cpu\" or \"cuda\".", call. = FALSE)
  }
  if (!output_device %in% c("cpu", "cuda")) {
    stop("`output_device` must be \"cpu\" or \"cuda\".", call. = FALSE)
  }
  if (!is.logical(fallback) || length(fallback) != 1L || is.na(fallback)) {
    stop("`fallback` must be TRUE or FALSE.", call. = FALSE)
  }
  if (fallback &&
      !(identical(requested_device, "auto") && identical(device, "cpu"))) {
    stop(
      "`fallback = TRUE` requires an automatic request that selected CPU.",
      call. = FALSE
    )
  }

  structure(
    list(
      requested_device = requested_device,
      device = device,
      backend = backend,
      selection_reason = selection_reason,
      fallback = fallback,
      output_device = output_device
    ),
    class = "cuda_stage"
  )
}

.validate_cuda_stages <- function(stages) {
  if (!is.list(stages) || !length(stages) || is.null(names(stages)) ||
      anyNA(names(stages)) || any(!nzchar(names(stages))) ||
      anyDuplicated(names(stages))) {
    stop(
      "Compute stages must be a non-empty, uniquely named list.",
      call. = FALSE
    )
  }
  result <- lapply(
    seq_along(stages),
    function(index) {
      stage <- stages[[index]]
      if (!is.list(stage) || !identical(names(stage), .cuda_stage_fields)) {
        stop(
          sprintf(
            "Compute stage `%s` does not follow the cudaverse stage schema.",
            names(stages)[[index]]
          ),
          call. = FALSE
        )
      }
      do.call(cuda_stage, unclass(stage))
    }
  )
  names(result) <- names(stages)
  result
}

.compute_device_from_stages <- function(stages) {
  stages <- .validate_cuda_stages(stages)
  devices <- unique(vapply(stages, `[[`, character(1), "device"))
  if (length(devices) == 1L) devices else "hybrid"
}

.object_compute_stages <- function(x) {
  if (inherits(x, "cuda_provenance")) {
    return(attr(x, "compute_stages", exact = TRUE))
  }
  if (is.list(x) && "compute_stages" %in% names(x)) {
    return(x[["compute_stages"]])
  }
  attr(x, "compute_stages", exact = TRUE)
}

.provenance_error <- function(message, subclass, expected, actual) {
  condition <- structure(
    list(
      message = message,
      call = NULL,
      expected = expected,
      actual = actual
    ),
    class = c(
      subclass,
      "cudaverse_provenance_error",
      "error",
      "condition"
    )
  )
  stop(condition)
}

.object_provenance_schema <- function(x) {
  if (inherits(x, "cuda_provenance")) {
    return(attr(x, "schema", exact = TRUE))
  }
  if (is.list(x) && "provenance_schema" %in% names(x)) {
    return(x[["provenance_schema"]])
  }
  attr(x, "provenance_schema", exact = TRUE)
}

.object_declared_compute_device <- function(x) {
  if (inherits(x, "cuda_provenance")) {
    return(attr(x, "compute_device", exact = TRUE))
  }
  if (is.list(x) && "compute_device" %in% names(x)) {
    return(x[["compute_device"]])
  }
  attr(x, "compute_device", exact = TRUE)
}

.validate_provenance_schema <- function(x) {
  declared_schema <- .object_provenance_schema(x)
  if (!is.null(declared_schema) &&
      !identical(declared_schema, .cudaverse_provenance_schema)) {
    .provenance_error(
      paste0(
        "Unsupported cudaverse provenance schema; expected `",
        .cudaverse_provenance_schema,
        "`."
      ),
      subclass = "cudaverse_provenance_schema_error",
      expected = .cudaverse_provenance_schema,
      actual = declared_schema
    )
  }
  invisible(declared_schema)
}

.validate_provenance_aggregate <- function(x, stages) {
  declared_compute_device <- .object_declared_compute_device(x)
  actual_compute_device <- .compute_device_from_stages(stages)
  if (!is.null(declared_compute_device) &&
      !identical(declared_compute_device, actual_compute_device)) {
    .provenance_error(
      paste0(
        "Declared cudaverse compute device does not match the recorded ",
        "compute stages."
      ),
      subclass = "cudaverse_provenance_aggregate_error",
      expected = actual_compute_device,
      actual = declared_compute_device
    )
  }
  invisible(actual_compute_device)
}

#' Inspect actual compute provenance
#'
#' Returns one row per computation stage. The table prevents an `"auto"`
#' request, a CUDA-aware kernel, or a hybrid pipeline from being mistaken for
#' end-to-end GPU execution.
#'
#' @param x A cudaverse result or a named list of `cuda_stage` records.
#' @return A `cuda_provenance` data frame with columns `stage`,
#'   `requested_device`, `device`, `backend`, `selection_reason`, `fallback`,
#'   and `output_device`. Its `schema` and `compute_device` attributes contain
#'   the contract version and aggregate actual compute device.
#' @export
#' @examples
#' x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
#' cuda_provenance(x)
cuda_provenance <- function(x) {
  .validate_provenance_schema(x)
  bare_stages <- is.list(x) && length(x) &&
      !is.null(names(x)) &&
      all(vapply(x, inherits, logical(1), "cuda_stage"))
  stages <- if (bare_stages) {
    x
  } else {
    .object_compute_stages(x)
  }
  if (is.null(stages)) {
    stop("`x` does not contain cudaverse compute-stage provenance.",
         call. = FALSE)
  }
  stages <- .validate_cuda_stages(stages)
  .validate_provenance_aggregate(x, stages)
  result <- data.frame(
    stage = names(stages),
    requested_device = vapply(
      stages, `[[`, character(1), "requested_device"
    ),
    device = vapply(stages, `[[`, character(1), "device"),
    backend = vapply(stages, `[[`, character(1), "backend"),
    selection_reason = vapply(
      stages, `[[`, character(1), "selection_reason"
    ),
    fallback = vapply(stages, `[[`, logical(1), "fallback"),
    output_device = vapply(
      stages, `[[`, character(1), "output_device"
    ),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
  attr(result, "schema") <- .cudaverse_provenance_schema
  attr(result, "compute_device") <- .compute_device_from_stages(stages)
  attr(result, "compute_stages") <- stages
  class(result) <- c("cuda_provenance", "data.frame")
  result
}

#' @export
print.cuda_provenance <- function(x, ...) {
  compute_device <- attr(x, "compute_device", exact = TRUE)
  schema <- attr(x, "schema", exact = TRUE)
  cat(sprintf(
    "<cuda_provenance schema=%s stages=%s compute=%s>\n",
    schema,
    nrow(x),
    compute_device
  ))
  printable <- x
  class(printable) <- "data.frame"
  print(printable, row.names = FALSE, ...)
  invisible(x)
}

#' @export
print.cuda_diagnostics <- function(x, ...) {
  cat(sprintf(
    paste0(
      "<cuda_diagnostics available=%s devices=%s torch=%s reason=%s>\n"
    ),
    x$cuda_available,
    x$cuda_device_count,
    if (is.na(x$torch_version)) "not installed" else x$torch_version,
    x$reason
  ))
  invisible(x)
}

#' @export
print.cuda_device_selection <- function(x, ...) {
  cat(sprintf(
    "<cuda_device_selection requested=%s selected=%s reason=%s fallback=%s>\n",
    x$requested_device,
    x$device,
    x$selection_reason,
    x$fallback
  ))
  invisible(x)
}
