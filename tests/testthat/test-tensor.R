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

test_that("reductions operate over one-based dimensions", {
  x <- cuda_tensor(matrix(1:6, 2, 3), device = "cpu")

  expect_equal(as.vector(to_cpu(tensor_sum(x))), 21)
  expect_equal(as.vector(to_cpu(tensor_mean(x, dim = 1))), c(1.5, 3.5, 5.5))
  expect_identical(tensor_shape(tensor_sum(x, dim = 2, keepdim = TRUE)),
                   c(2L, 1L))
})

test_that("broadcasting follows trailing-dimension rules", {
  x <- cuda_tensor(1:3, device = "cpu")
  result <- to_cpu(tensor_broadcast_to(x, c(2, 3)))

  expect_identical(dim(result), c(2L, 3L))
  expect_equal(result[1, ], c(1, 2, 3))
  expect_equal(result[2, ], c(1, 2, 3))
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
})
