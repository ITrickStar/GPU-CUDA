#pragma warning(disable : 4244)

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#include "gpuErrchk.h"

#define idx (threadIdx.x + blockIdx.x * blockDim.x)

__global__ void kernelSum(float *input_gpu, int n) {
  if (idx < n) input_gpu[idx] += idx;
}
__global__ void kernelidx() {
  printf("I am from %u block, %u thread (global index: %u)\n", blockIdx.x,
         threadIdx.x, idx);
}
__global__ void kernelHello() { printf("Hello, world!\n"); }

__host__ int main() {
  int count, dev;

  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, 0);
  std::cout << props.name << ": " << props.major << "." << props.minor
             << std::endl;


  gpuErrchk(cudaGetDeviceCount(&count));
  gpuErrchk(cudaGetDevice(&dev));
  printf("all_dev: %i curr_dev: %i\n", count, dev);

  const int block_size = 256;
  int n = 1025;
  int num_blocks = (n + block_size - 1) / block_size;

  kernelidx<<<num_blocks, block_size>>>();
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());
  printf("\n");

  float *arr = new float[n], *arr_gpu;
  gpuErrchk(cudaMalloc((void **)&arr_gpu, n * sizeof(float)));

  for (int i = 0; i < n; i++) arr[i] = i;
  gpuErrchk(
      cudaMemcpyAsync(arr_gpu, arr, n * sizeof(float), cudaMemcpyHostToDevice));

  kernelSum<<<num_blocks, block_size>>>(arr_gpu, n);
  gpuErrchk(cudaGetLastError());
  gpuErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < n; i++) printf("%f ", arr[i]);
  printf("\n");

  gpuErrchk(
      cudaMemcpy(arr, arr_gpu, n * sizeof(float), cudaMemcpyDeviceToHost));

  for (int i = 0; i < n; i++) printf("%f ", arr[i]);
  printf("\n");

  cudaFree(arr_gpu);

  return 0;
}
