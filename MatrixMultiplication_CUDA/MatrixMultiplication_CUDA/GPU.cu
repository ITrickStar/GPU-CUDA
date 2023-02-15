#include "GPU.cuh"

#define idx (blockIdx.x * blockDim.x + threadIdx.x)
#define idy (blockIdx.y * blockDim.y + threadIdx.y)
__global__ void kernel_naive(float* A, float* B, float* C, int M, int N,
                             int K) {
  for (int i = 0; i < N; i++) {
    C[idx * K + idy] += A[i * M + idx] * B[i * K + idy];
  }
}

float* GPUMultiNaive(float* A, float* B, int M, int N, int K,
                     std::vector<double>* attmpts) {
  float* C = MatrixZeros(M, K);
  TMatrix(A, M, N);

  float *A_gpu, *B_gpu, *C_gpu;

  gpuErrchk(cudaMalloc((void**)&A_gpu, M * N * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&B_gpu, N * K * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&C_gpu, M * K * sizeof(float)));

  gpuErrchk(
      cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(B_gpu, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(C_gpu, C, M * K * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 num_blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (K + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  double start, end;
  start = omp_get_wtime();

  kernel_naive<<<num_blocks, block_size>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  gpuErrchk(cudaDeviceSynchronize());

  end = omp_get_wtime();
  if (attmpts != nullptr) (*attmpts).push_back(end - start);

  gpuErrchk(cudaGetLastError());

  gpuErrchk(
      cudaMemcpy(C, C_gpu, M * K * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  TMatrix(A, N, M);

  return C;
}

#define t_x threadIdx.x
#define t_y threadIdx.y
__global__ void kernel_optimized(float* A, float* B, float* C, int M, int N,
                                 int K) {
  int A_start = blockIdx.x * BLOCK_SIZE * N;
  int B_start = blockIdx.y * BLOCK_SIZE * N;

  __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];

  float sum = 0.0f;
  for (int i = A_start, j = B_start, k = 0; k < N;
       i += BLOCK_SIZE, j += BLOCK_SIZE, k += BLOCK_SIZE) {
    A_shared[t_x][t_y] = A[i + t_y * N + t_x];
    B_shared[t_x][t_y] = B[j + t_y * N + t_x];

    __syncthreads();
    for (int i = 0; i < BLOCK_SIZE; i++) {
      sum += A_shared[i][t_x] * B_shared[i][t_y];
    }
    __syncthreads();
  }

  int C_start = blockIdx.x * BLOCK_SIZE * K + blockIdx.y * BLOCK_SIZE;
  C[C_start + t_x * K + t_y] = sum;
}

float* GPUMultiOptimized(float* A, float* B, int M, int N, int K,
                         std::vector<double>* times) {
  float* C = MatrixZeros(M, K);
  TMatrix(B, N, K);

  float *A_gpu, *B_gpu, *C_gpu;

  gpuErrchk(cudaMalloc((void**)&A_gpu, M * N * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&B_gpu, N * K * sizeof(float)));
  gpuErrchk(cudaMalloc((void**)&C_gpu, M * K * sizeof(float)));

  gpuErrchk(
      cudaMemcpy(A_gpu, A, M * N * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(B_gpu, B, N * K * sizeof(float), cudaMemcpyHostToDevice));
  gpuErrchk(
      cudaMemcpy(C_gpu, C, M * K * sizeof(float), cudaMemcpyHostToDevice));

  dim3 block_size(BLOCK_SIZE, BLOCK_SIZE, 1);
  dim3 num_blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (K + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);

  double start, end;
  start = omp_get_wtime();

  kernel_optimized<<<num_blocks, block_size>>>(A_gpu, B_gpu, C_gpu, M, N, K);
  gpuErrchk(cudaDeviceSynchronize());

  end = omp_get_wtime();
  if (times != nullptr) (*times).push_back(end - start);

  gpuErrchk(cudaGetLastError());

  gpuErrchk(
      cudaMemcpy(C, C_gpu, M * K * sizeof(float), cudaMemcpyDeviceToHost));

  cudaFree(A_gpu);
  cudaFree(B_gpu);
  cudaFree(C_gpu);
  TMatrix(B, K, N);

  return C;
}
