test_that("device diagnostics and explicit CPU selection are structured", {
  diagnostics <- cuda_diagnostics()
  expect_s3_class(diagnostics, "cuda_diagnostics")
  expect_type(diagnostics$torch_installed, "logical")
  expect_length(diagnostics$torch_installed, 1L)
  expect_type(diagnostics$cuda_available, "logical")
  expect_length(diagnostics$cuda_available, 1L)
  expect_true(
    diagnostics$reason %in%
      c(
        "torch_not_installed",
        "backend_error",
        "cuda_available",
        "cuda_unavailable"
      )
  )

  selection <- cuda_select_device("cpu")
  expect_s3_class(selection, "cuda_device_selection")
  expect_identical(selection$requested_device, "cpu")
  expect_identical(selection$device, "cpu")
  expect_identical(selection$selection_reason, "explicit_cpu")
  expect_false(selection$fallback)
  expect_null(selection$diagnostics)
})

test_that("device-count probe errors make CUDA unavailable", {
  testthat::local_mocked_bindings(
    .cuda_torch_installed = function() TRUE,
    .cuda_torch_version = function() "test",
    .cuda_torch_is_available = function() TRUE,
    .cuda_torch_device_count = function() {
      stop("device-count probe failed")
    }
  )

  diagnostics <- cuda_diagnostics()
  expect_false(diagnostics$cuda_available)
  expect_identical(diagnostics$cuda_device_count, NA_integer_)
  expect_identical(diagnostics$reason, "backend_error")
  expect_match(diagnostics$detection_error, "device-count probe failed")
})

test_that("invalid and empty device counts cannot select CUDA", {
  probe_count <- NA_integer_
  testthat::local_mocked_bindings(
    .cuda_torch_installed = function() TRUE,
    .cuda_torch_version = function() "test",
    .cuda_torch_is_available = function() TRUE,
    .cuda_torch_device_count = function() probe_count
  )

  for (value in list(NA_integer_, c(1L, 2L), -1L, 1.5)) {
    probe_count <- value
    diagnostics <- cuda_diagnostics()
    expect_false(diagnostics$cuda_available)
    expect_identical(diagnostics$cuda_device_count, NA_integer_)
    expect_identical(diagnostics$reason, "backend_error")
    expect_match(diagnostics$detection_error, "invalid value")
  }

  probe_count <- 0L
  diagnostics <- cuda_diagnostics()
  expect_false(diagnostics$cuda_available)
  expect_identical(diagnostics$cuda_device_count, 0L)
  expect_identical(diagnostics$reason, "cuda_unavailable")
  expect_null(diagnostics$detection_error)
})

test_that("automatic CPU fallback and strict CUDA errors retain their reason", {
  unavailable <- structure(
    list(
      torch_installed = FALSE,
      torch_version = NA_character_,
      cuda_available = FALSE,
      cuda_device_count = 0L,
      reason = "torch_not_installed",
      detection_error = NULL
    ),
    class = "cuda_diagnostics"
  )
  testthat::local_mocked_bindings(
    cuda_diagnostics = function() unavailable
  )

  automatic <- cuda_select_device("auto")
  expect_identical(automatic$requested_device, "auto")
  expect_identical(automatic$device, "cpu")
  expect_identical(automatic$selection_reason, "torch_not_installed")
  expect_true(automatic$fallback)

  condition <- tryCatch(
    cuda_select_device("cuda"),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_cuda_unavailable")
  expect_match(conditionMessage(condition), "CUDA is unavailable")
  expect_identical(condition$diagnostics$reason, "torch_not_installed")
})

test_that("compute stages validate and summarize provenance", {
  stages <- list(
    distance = cuda_stage(
      requested_device = "cuda",
      device = "cuda",
      backend = "torch",
      selection_reason = "explicit_cuda",
      output_device = "cpu"
    ),
    ordering = cuda_stage(
      requested_device = "fixed-cpu",
      device = "cpu",
      backend = "base",
      selection_reason = "algorithm_cpu_only"
    )
  )
  provenance <- cuda_provenance(stages)

  expect_s3_class(provenance, "cuda_provenance")
  expect_identical(
    names(provenance),
    c(
      "stage",
      "requested_device",
      "device",
      "backend",
      "selection_reason",
      "fallback",
      "output_device"
    )
  )
  expect_identical(provenance$stage, c("distance", "ordering"))
  expect_identical(attr(provenance, "schema"), "cudaverse-stage/1")
  expect_identical(attr(provenance, "compute_device"), "hybrid")

  expect_error(
    cuda_stage(
      requested_device = "cuda",
      device = "cpu",
      backend = "base",
      selection_reason = "invalid",
      fallback = TRUE
    ),
    "automatic request"
  )
  expect_error(cuda_provenance(list()), "does not contain")
})

test_that("declared provenance schema and aggregate are strictly validated", {
  x <- cuda_tensor(1:3, device = "cpu")

  wrong_schema <- x
  wrong_schema$provenance_schema <- "cudaverse-stage/99"
  condition <- tryCatch(
    cuda_provenance(wrong_schema),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_provenance_schema_error")
  expect_s3_class(condition, "cudaverse_provenance_error")
  expect_identical(condition$expected, "cudaverse-stage/1")
  expect_identical(condition$actual, "cudaverse-stage/99")

  future_schema <- wrong_schema
  future_schema$compute_stages <- list(
    future_stage = list(device = "accelerator")
  )
  condition <- tryCatch(
    cuda_provenance(future_schema),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_provenance_schema_error")
  expect_identical(condition$actual, "cudaverse-stage/99")
  expect_false(grepl(
    "does not follow the cudaverse stage schema",
    conditionMessage(condition),
    fixed = TRUE
  ))

  wrong_aggregate <- x
  wrong_aggregate$compute_device <- "cuda"
  condition <- tryCatch(
    cuda_provenance(wrong_aggregate),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_provenance_aggregate_error")
  expect_s3_class(condition, "cudaverse_provenance_error")
  expect_identical(condition$expected, "cpu")
  expect_identical(condition$actual, "cuda")
})

test_that("bare stage lists remain valid provenance inputs", {
  stages <- list(
    work = cuda_stage(
      requested_device = "fixed-cpu",
      device = "cpu",
      backend = "base",
      selection_reason = "algorithm_cpu_only"
    )
  )

  provenance <- cuda_provenance(stages)
  expect_identical(provenance$stage, "work")
  expect_identical(attr(provenance, "schema"), "cudaverse-stage/1")
  expect_identical(attr(provenance, "compute_device"), "cpu")

  attr(provenance, "schema") <- "cudaverse-stage/99"
  condition <- tryCatch(
    cuda_provenance(provenance),
    error = identity
  )
  expect_s3_class(condition, "cudaverse_provenance_schema_error")
})

test_that("tensor provenance separates requests from actual execution", {
  explicit <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
  provenance <- cuda_provenance(explicit)
  expect_identical(provenance$stage, "tensor_materialization")
  expect_identical(provenance$requested_device, "cpu")
  expect_identical(provenance$device, "cpu")
  expect_identical(provenance$backend, "base")
  expect_identical(provenance$selection_reason, "explicit_cpu")
  expect_false(provenance$fallback)
  expect_identical(provenance$output_device, "cpu")
  expect_identical(explicit$compute_device, "cpu")
})

test_that("CPU float32 storage uses IEEE single-precision rounding", {
  source <- matrix(c(1 / 3, pi, -sqrt(2), 1e-20), 2, 2)
  expected <- array(
    readBin(
      writeBin(as.numeric(source), raw(), size = 4L),
      what = double(),
      n = length(source),
      size = 4L
    ),
    dim = dim(source)
  )
  actual <- cuda_tensor(source, device = "cpu", dtype = "float32")

  expect_identical(to_cpu(actual), expected)
  expect_false(identical(as.numeric(to_cpu(actual)), as.numeric(source)))
  expect_identical(
    as.numeric(to_cpu(
      cuda_tensor(1e100, device = "cpu", dtype = "float32")
    )),
    Inf
  )
})

test_that("tensor operations identify the stage that actually ran", {
  x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
  y <- cuda_tensor(matrix(1:6, 3, 2), device = "cpu")

  expect_identical(
    cuda_provenance(tensor_matmul(x, y))$stage,
    "matrix_multiply"
  )
  expect_identical(cuda_provenance(tensor_sum(x))$stage, "sum")
  expect_identical(cuda_provenance(tensor_mean(x))$stage, "mean")
  expect_identical(
    cuda_provenance(tensor_reshape(x, c(3, 2)))$stage,
    "reshape"
  )
  broadcast <- tensor_broadcast_to(
    cuda_tensor(1:3, "cpu"),
    c(2, 3)
  )
  expect_identical(cuda_provenance(broadcast)$stage, "broadcast")
  expect_identical(cuda_provenance(x + 1)$stage, "arithmetic")
  expect_identical(cuda_provenance(x[1, ])$stage, "subset")
  x[1, 1] <- 10
  expect_identical(cuda_provenance(x)$stage, "replacement")
  expect_identical(cuda_provenance(t(x))$stage, "transpose")
})
