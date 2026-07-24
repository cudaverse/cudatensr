library(cudatensr)

set.seed(1)
x <- matrix(rnorm(1e6), 1000)
y <- matrix(rnorm(1e6), 1000)

base_time <- system.time(base_result <- x %*% y)
tensor_time <- system.time({
  tensor_result <- tensor_matmul(
    cuda_tensor(x),
    cuda_tensor(y)
  )
  tensor_result <- to_cpu(tensor_result)
})

stopifnot(isTRUE(all.equal(base_result, tensor_result, tolerance = 1e-6)))
print(rbind(base = base_time, cudatensr = tensor_time))
