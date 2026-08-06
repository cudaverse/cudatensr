library(cudaverse)

set.seed(1)
x <- matrix(rnorm(5000 * 100), 5000, 100)

base_pca_time <- system.time(stats::prcomp(x, rank. = 20))
cuda_pca_time <- system.time(fit <- cuda_pca(x, 20))
knn_time <- system.time(cuda_knn(fit$x, 15, batch_size = 256))

print(rbind(
  base_pca = base_pca_time,
  cudaverse_pca = cuda_pca_time,
  cudaverse_knn = knn_time
))
