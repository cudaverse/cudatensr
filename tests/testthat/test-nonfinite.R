test_that("floating tensors preserve IEEE non-finite values", {
  values <- c(1, Inf, -Inf, NaN, NA_real_)
  tensor <- cuda_tensor(values, device = "cpu", dtype = "float64")
  actual <- as.numeric(to_cpu(tensor))

  expect_identical(actual[1:3], values[1:3])
  expect_true(is.nan(actual[[4]]))
  expect_true(is.na(actual[[5]]))
  expect_false(is.nan(actual[[5]]))
})

test_that("derived non-finite arithmetic is consistent on the CPU backend", {
  numerator <- cuda_tensor(c(1, 0, -1), device = "cpu")
  denominator <- cuda_tensor(c(0, 0, 0), device = "cpu")
  result <- numerator / denominator

  expect_identical(as.numeric(to_cpu(result)), c(Inf, NaN, -Inf))
  expect_identical(result$dtype, "float64")
  expect_identical(cuda_provenance(result)$stage, "arithmetic")
})

test_that("floating replacement accepts non-finite values", {
  tensor <- cuda_tensor(c(1, 2, 3), device = "cpu")
  tensor[1:2] <- c(Inf, NA_real_)

  expect_identical(as.numeric(to_cpu(tensor)), c(Inf, NA_real_, 3))
})

test_that("integer tensors reject non-finite values clearly", {
  expect_error(
    cuda_tensor(NA_integer_, device = "cpu", dtype = "integer"),
    "represented exactly"
  )
  expect_error(
    cuda_tensor(Inf, device = "cpu", dtype = "integer"),
    "represented exactly"
  )

  tensor <- cuda_tensor(1:3, device = "cpu", dtype = "integer")
  expect_error(
    tensor[1] <- Inf,
    "represented exactly"
  )
})
