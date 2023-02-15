#include <omp.h>
#include <stdio.h>

#include <iomanip>

#include "CPU.h"
#include "GPU.cuh"

using namespace std;

void PrintGPUSpecs() {
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  cout << "Device Properties:" << endl;
  cout << "  " << deviceProp.name << ": " << deviceProp.major << "."
       << deviceProp.minor << endl;
  cout << "  Global memory:   " << deviceProp.totalGlobalMem << "mb" << endl;
  cout << "  Shared memory:   " << deviceProp.sharedMemPerBlock << "kb" << endl;
  cout << "  Constant memory: " << deviceProp.totalConstMem << "kb" << endl;
  cout << "  Block registers: " << deviceProp.regsPerBlock << endl;

  cout << "  Warp size:       " << deviceProp.warpSize << endl;
  cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << endl;
  cout << "  Max block dimensions: [" << deviceProp.maxThreadsDim[0] << ", "
       << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2]
       << "]" << endl;
  cout << "  Max grid dimensions:  [" << deviceProp.maxGridSize[0] << ", "
       << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << "]"
       << endl
       << endl;
}

void PrintResults(double SEQ, double OMP, double GPU_Naive, double GPU_Opt) {
  cout << "Technology" << '\t' << "Time(sec.)" << '\t' << "Acceleration"
       << endl;
  cout << "SEQ" << '\t' << '\t' << SEQ << ' ' << '\t' << SEQ / SEQ << endl;
  cout << "OMP" << '\t' << '\t' << OMP << '\t' << '\t' << SEQ / OMP << endl;
  cout << "GPU Naive" << '\t' << GPU_Naive << '\t' << '\t' << SEQ / GPU_Naive
       << endl;
  cout << "GPU Opt" << '\t' << '\t' << GPU_Opt << '\t' << '\t' << SEQ / GPU_Opt
       << endl;
}

int main() {
  cout << fixed << setprecision(5);
  PrintGPUSpecs();

  int M = 100 * BLOCK_SIZE, N = 110 * BLOCK_SIZE, K = 120 * BLOCK_SIZE;

  cout << "Matrix A is " << '[' << M << ", " << N << ']' << endl;
  cout << "Matrix B is " << '[' << N << ", " << K << ']' << endl;
  cout << "Matrix C is " << '[' << M << ", " << K << ']' << endl << endl;

  float* A = Matrix(M, N);
  float* B = Matrix(N, K);

  vector<double> attmpts;
  int iter = 10;

  double start, end;
  double SEQ, OMP, GPU_Naive, GPU_Opt;
  float *resSEQ, *resOMP, *resGPUNaive, *resGPUOpt;

  for (int i = 0; i < iter; i++) {
    start = omp_get_wtime();
    resSEQ = SEQMulti(A, B, M, N, K);
    end = omp_get_wtime();
    attmpts.push_back(end - start);
  }
  SEQ = AverageTime(attmpts);
  attmpts.clear();
  cout << "SEQ = " << SEQ << endl;

  for (int i = 0; i < iter; i++) {
    start = omp_get_wtime();
    resOMP = OMPMulti(A, B, M, N, K);
    end = omp_get_wtime();
    attmpts.push_back(end - start);
  }
  OMP = AverageTime(attmpts);
  attmpts.clear();
  cout << "OMP = " << OMP << endl;

  for (int i = 0; i < iter; i++) {
    resGPUNaive = GPUMultiNaive(A, B, M, N, K, &attmpts);
  }
  GPU_Naive = AverageTime(attmpts);
  attmpts.clear();
  cout << "GPUMultiNaive = " << GPU_Naive << endl;

  for (int i = 0; i < iter; i++) {
    resGPUOpt = GPUMultiOptimized(A, B, M, N, K, &attmpts);
  }
  GPU_Opt = AverageTime(attmpts);
  attmpts.clear();
  cout << "GPUMultiOptimized = " << GPU_Opt << endl;

  bool OMP_Correct = CompareMatrix(resSEQ, resOMP, M, K);
  bool GPUNaiveOptimized_Correct = CompareMatrix(resSEQ, resGPUNaive, M, K);
  bool GPUOptimized_Correct = CompareMatrix(resSEQ, resGPUOpt, M, K);

  cout << "Matrix Comparison: " << OMP_Correct << " "
       << GPUNaiveOptimized_Correct << " " << GPUOptimized_Correct << endl
       << endl;

  // PrintMatrix(resSEQ, M, K);
  // PrintMatrix(resGPUNaive, M, K);
  // PrintMatrix(resGPUOpt, M, K);

  PrintResults(SEQ, OMP, GPU_Naive, GPU_Opt);

  delete[] A, B, resSEQ, resOMP, resGPUNaive, resGPUOpt;

  return 0;
}
