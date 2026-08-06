library(cudaverse)
library(Matrix)

set.seed(1)
x <- rsparsematrix(10000, 2000, density = 0.001)
y <- matrix(rnorm(2000 * 20), 2000, 20)

matrix_time <- system.time(expected <- x %*% y)
cuda_object <- cuda_sparse(x)
cudaverse_time <- system.time({
  actual <- to_cpu(sparse_matmul_dense(cuda_object, y))
})

stopifnot(isTRUE(all.equal(as.matrix(expected), actual, tolerance = 1e-6)))
print(rbind(Matrix = matrix_time, cudaverse = cudaverse_time))
