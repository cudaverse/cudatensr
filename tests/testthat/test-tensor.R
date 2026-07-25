test_that("CPU tensors preserve shape, dtype, and values", {
  x <- cuda_tensor(matrix(1:6, 2), device = "cpu")

  expect_s3_class(x, "cudatensor")
  expect_identical(tensor_shape(x), c(2L, 3L))
  expect_identical(unname(tensor_device(x)), c("cpu", "base"))
  expect_equal(to_cpu(x), matrix(as.integer(1:6), 2))
})

test_that("matrix multiplication matches base R", {
  a <- matrix(1:6, 2, 3)
  b <- matrix(1:6, 3, 2)
  result <- tensor_matmul(
    cuda_tensor(a, device = "cpu"),
    cuda_tensor(b, device = "cpu")
  )

  expect_equal(to_cpu(result), a %*% b)
  expect_identical(tensor_shape(result), c(2L, 2L))
})

test_that("matrix multiplication promotes mixed and integer dtypes safely", {
  mixed <- tensor_matmul(
    cuda_tensor(matrix(1:4, 2), device = "cpu"),
    matrix(c(0.5, 1), 2, 1)
  )
  large <- cuda_tensor(matrix(50000L, 1), device = "cpu") %*%
    cuda_tensor(matrix(50000L, 1), device = "cpu")

  expect_equal(to_cpu(mixed), matrix(c(3.5, 5), 2, 1))
  expect_identical(mixed$dtype, "float64")
  expect_equal(as.numeric(to_cpu(large)), 2500000000)
  expect_false(anyNA(to_cpu(large)))
  expect_identical(large$dtype, "float64")
})

test_that("reductions operate over one-based dimensions", {
  x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")

  expect_equal(as.vector(to_cpu(tensor_sum(x))), 21)
  expect_equal(as.vector(to_cpu(tensor_mean(x, dim = 1))), c(1.5, 3.5, 5.5))
  expect_identical(tensor_shape(tensor_sum(x, dim = 2, keepdim = TRUE)),
                   c(2L, 1L))
})

test_that("integer reductions do not overflow", {
  x <- cuda_tensor(c(.Machine$integer.max, 1L), device = "cpu")
  result <- tensor_sum(x)

  expect_equal(as.numeric(to_cpu(result)), .Machine$integer.max + 1)
  expect_identical(result$dtype, "float64")
})

test_that("broadcasting follows trailing-dimension rules", {
  x <- cuda_tensor(1:3, device = "cpu")
  result <- to_cpu(tensor_broadcast_to(x, c(2, 3)))

  expect_identical(dim(result), c(2L, 3L))
  expect_equal(result[1, ], c(1, 2, 3))
  expect_equal(result[2, ], c(1, 2, 3))
})

test_that("arithmetic operators broadcast and promote without truncation", {
  x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")
  column_values <- cuda_tensor(c(0.5, 1, 1.5), device = "cpu")

  product <- x * column_values
  shifted <- 0.5 + x

  expect_equal(
    to_cpu(product),
    matrix(1:6, 2, 3) *
      matrix(rep(c(0.5, 1, 1.5), each = 2), 2, 3)
  )
  expect_equal(to_cpu(shifted), matrix(1:6, 2, 3) + 0.5)
  expect_identical(product$dtype, "float64")
  expect_identical(shifted$dtype, "float64")
})

test_that("integer dtype conversion rejects lossy values", {
  expect_error(
    cuda_tensor(0.5, device = "cpu", dtype = "integer"),
    "cannot be represented exactly"
  )
  expect_error(
    cuda_tensor(.Machine$integer.max + 1, device = "cpu", dtype = "integer"),
    "cannot be represented exactly"
  )
})

test_that("invalid tensor operations fail clearly", {
  expect_error(cuda_tensor(c(1, NA), device = "cpu"), "finite")
  expect_error(
    tensor_matmul(
      cuda_tensor(matrix(1:4, 2), device = "cpu"),
      cuda_tensor(matrix(1:6, 3), device = "cpu")
    ),
    "not conformable"
  )
  expect_error(
    tensor_broadcast_to(cuda_tensor(1:3, device = "cpu"), c(2, 2)),
    "not compatible"
  )
  expect_error(
    cuda_tensor(1:3, device = "cpu") +
      cuda_tensor(matrix(1:4, 2), device = "cpu"),
    "not compatible"
  )
  expect_error(
    cuda_tensor(1:3, device = "cpu") == 1,
    "not supported"
  )
})
