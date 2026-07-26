# Package index

## Create, inspect, and move tensors

- [`cuda_tensor()`](https://cudaverse.github.io/cudatensr/reference/cuda_tensor.md)
  : Create a GPU-aware tensor
- [`cuda_available()`](https://cudaverse.github.io/cudatensr/reference/cuda_available.md)
  : Detect a usable CUDA backend
- [`cuda_diagnostics()`](https://cudaverse.github.io/cudatensr/reference/cuda_diagnostics.md)
  : Diagnose the optional CUDA runtime
- [`cuda_select_device()`](https://cudaverse.github.io/cudatensr/reference/cuda_select_device.md)
  : Select a computation device without hiding fallback
- [`cuda_provenance()`](https://cudaverse.github.io/cudatensr/reference/cuda_provenance.md)
  : Inspect actual compute provenance
- [`cuda_stage()`](https://cudaverse.github.io/cudatensr/reference/cuda_stage.md)
  : Record one compute stage
- [`tensor_shape()`](https://cudaverse.github.io/cudatensr/reference/tensor_shape.md)
  : Inspect tensor shape
- [`dimnames(`*`<cudatensor>`*`)`](https://cudaverse.github.io/cudatensr/reference/dimnames.cudatensor.md)
  : Inspect tensor dimension labels
- [`tensor_device()`](https://cudaverse.github.io/cudatensr/reference/tensor_device.md)
  : Inspect tensor device and backend
- [`to_cpu()`](https://cudaverse.github.io/cudatensr/reference/to_cpu.md)
  : Transfer tensor data to base R
- [`to_device()`](https://cudaverse.github.io/cudatensr/reference/to_device.md)
  : Transfer a tensor to a device

## Transform and compute

- [`Ops(`*`<cudatensor>`*`)`](https://cudaverse.github.io/cudatensr/reference/cudatensor-operators.md)
  [`` `%*%`( ``*`<cudatensor>`*`)`](https://cudaverse.github.io/cudatensr/reference/cudatensor-operators.md)
  : Arithmetic operators for GPU-aware tensors
- [`` `[`( ``*`<cudatensor>`*`)`](https://cudaverse.github.io/cudatensr/reference/cudatensor-subset.md)
  [`` `[<-`( ``*`<cudatensor>`*`)`](https://cudaverse.github.io/cudatensr/reference/cudatensor-subset.md)
  : Subset and replace tensor values
- [`tensor_reshape()`](https://cudaverse.github.io/cudatensr/reference/tensor_reshape.md)
  : Reshape a tensor without changing its values
- [`tensor_broadcast_to()`](https://cudaverse.github.io/cudatensr/reference/tensor_broadcast_to.md)
  : Broadcast a tensor to a compatible shape
- [`tensor_matmul()`](https://cudaverse.github.io/cudatensr/reference/tensor_matmul.md)
  : Matrix multiplication for tensors
- [`tensor_sum()`](https://cudaverse.github.io/cudatensr/reference/tensor_sum.md)
  [`tensor_mean()`](https://cudaverse.github.io/cudatensr/reference/tensor_sum.md)
  : Tensor reductions
