#include <omp.h>
#include <stdio.h>

#include "ArrayFeatures.h"
#include "GPU.cuh"

#define idx (threadIdx.x + blockIdx.x * blockDim.x)

__global__ void kernelF(float a, float* x, int incx, float* y, int incy,
                        int threads_num, int n) {
  if (idx * incx >= n || idx * incy >= n) return;
  y[idx * incy] = y[idx * incy] + a * x[idx * incx];
}

float* saxpy_gpu(int threads_num, int block_size, int n, float a, float* x,
                 int incx, float* y, int incy, std::vector<double>* times) {
  int num_blocks = (n + block_size - 1) / block_size;

  float *x_gpu, *y_gpu;

  gpuErrchk(cudaMalloc((void**)&x_gpu, n * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&y_gpu, n * sizeof(float)));
  gpuErrchk(cudaMemcpy(x_gpu, x, n * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(y_gpu, y, n * sizeof(float), cudaMemcpyHostToDevice));

  double start, end;
  start = omp_get_wtime();
  kernelF<<<num_blocks, block_size>>>(a, x_gpu, incx, y_gpu, incy, threads_num,
                                      n);

  gpuErrchk(cudaDeviceSynchronize());

  end = omp_get_wtime();
  if (times != nullptr) (*times).push_back(end - start);

  gpuErrchk(cudaMemcpy(y, y_gpu, n * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(x_gpu);
  cudaFree(y_gpu);

  return y;
}

__global__ void kernelD(double a, double* x, int incx, double* y, int incy,
                        int threads_num, int n) {
  if (idx * incx >= n || idx * incy >= n) return;
  y[idx * incy] = y[idx * incy] + a * x[idx * incx];
}

double* daxpy_gpu(int threads_num, int block_size, int n, double a, double* x,
                  int incx, double* y, int incy, std::vector<double>* attmpts) {
  int num_blocks = (n + block_size - 1) / block_size;

  double *x_gpu, *y_gpu;

  gpuErrchk(cudaMalloc((void**)&x_gpu, n * sizeof(double)));
  gpuErrchk(cudaMalloc((void**)&y_gpu, n * sizeof(double)));
  gpuErrchk(cudaMemcpy(x_gpu, x, n * sizeof(double), cudaMemcpyHostToDevice));
  gpuErrchk(cudaMemcpy(y_gpu, y, n * sizeof(double), cudaMemcpyHostToDevice));

  double start, end;
  start = omp_get_wtime();
  kernelD<<<num_blocks, block_size>>>(a, x_gpu, incx, y_gpu, incy, threads_num,
                                      n);

  gpuErrchk(cudaDeviceSynchronize());
  end = omp_get_wtime();
  if (attmpts != nullptr) {
    (*attmpts).push_back(end - start);
  }

  gpuErrchk(cudaMemcpy(y, y_gpu, n * sizeof(double), cudaMemcpyDeviceToHost));

  cudaFree(x_gpu);
  cudaFree(y_gpu);

  return y;
}
