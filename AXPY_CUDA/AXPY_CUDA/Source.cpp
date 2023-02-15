#include <omp.h>
#include <stdio.h>

#include <iostream>
#include <vector>

#include "ArrayFeatures.h"
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
  cout << "  Max block dimensions: [ " << deviceProp.maxThreadsDim[0] << ", "
       << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2]
       << " ]" << endl;
  cout << "  Max grid dimensions:  [ " << deviceProp.maxGridSize[0] << ", "
       << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << " ]"
       << endl
       << endl
       << endl;
}

int BestBlockSize(const std::vector<int> block_sizes, const int threads_num,
                  const int iterations, const int n, const int incx,
                  const int incy, const double a) {
  std::vector<double> attmpts, avg_time;

  cout << "Best Block Size:" << endl;

  float* x = CreateArray<float>(n);
  float* y = CreateArray<float>(n);
  for (int i = 0; i < block_sizes.size(); i++) {
    for (int j = 0; j < iterations; j++) {
      saxpy_gpu(threads_num, block_sizes[i], n, a, x, incx, y, incy, &avg_time);
    }

    attmpts.push_back(AverageTime(avg_time));
    cout << "  Blocks: " << block_sizes[i];
    cout << '\t' << "Average time: " << AverageTime(avg_time) << endl;
  }
  int best_block_size = block_sizes[FindMinElementIdx(attmpts)];
  cout << "  Best block size: " << best_block_size << endl << endl;

  delete[] x;
  delete[] y;

  return best_block_size;
}

void PrintResults(int n, double CPU_f, double OMP_f, double GPU_f, double CPU_d,
                double OMP_d, double GPU_d) {
  cout << "Results for SAXPY (float): " << endl;
  cout << "Technology" << '\t' << "Time" << '\t' << '\t' << "Acceleration"
       << endl;
  cout << "CPU" << '\t' << '\t' << CPU_f << '\t' << CPU_f / CPU_f << endl;
  cout << "OMP" << '\t' << '\t' << OMP_f << '\t' << CPU_f / OMP_f << endl;
  cout << "GPU" << '\t' << '\t' << GPU_f << '\t' << CPU_f / GPU_f << endl;
  cout << endl;
  cout << "Results for DAXPY (double): " << endl;
  cout << "Technology" << '\t' << "Time" << '\t' << '\t' << "Acceleration"
       << endl;
  cout << "CPU" << '\t' << '\t' << CPU_d << '\t' << CPU_d / CPU_d << endl;
  cout << "OMP" << '\t' << '\t' << OMP_d << '\t' << CPU_d / OMP_d << endl;
  cout << "GPU" << '\t' << '\t' << GPU_d << '\t' << CPU_d / GPU_d << endl;
}

int main() {
  PrintGPUSpecs();

  const int threads_num = 100000000;
  const int iter = 10;

  int n = threads_num;
  const int incx = 2, incy = 3;
  const double a = 0.5;

  cout << "Array Size = " << n << endl;

  const std::vector<int> block_sizes{8, 16, 32, 64, 128, 256};
  int block_size =
      BestBlockSize(block_sizes, threads_num, iter, n, incx, incy, a);

  double start, end;
  double CPU_f, OMP_f, GPU_f;
  std::vector<double> attmpts;

  // SAXPY //

  float* x = CreateArray<float>(n);
  float* y = CreateArray<float>(n);

  float *resSEQ, *resOMP, *resGPU;

  // CPU //
  // SEQ //
  for (int i = 0; i < iter; i++) {
    start = omp_get_wtime();
    resSEQ = saxpy_seq(n, a, x, incx, y, incy);
    end = omp_get_wtime();
    attmpts.push_back(end - start);
  }
  CPU_f = AverageTime(attmpts);

  attmpts.clear();
  x = CreateArray<float>(n);
  y = CreateArray<float>(n);

  // OMP
  for (int i = 0; i < iter; i++) {
    start = omp_get_wtime();
    resOMP = saxpy_omp(n, a, x, incx, y, incy);
    end = omp_get_wtime();
    attmpts.push_back(end - start);
  }
  OMP_f = AverageTime(attmpts);

  x = CreateArray<float>(n);
  y = CreateArray<float>(n);
  attmpts.clear();

  // GPU //
  for (int i = 0; i < iter; i++) {
    resGPU =
        saxpy_gpu(threads_num, block_size, n, a, x, incx, y, incy, &attmpts);
  }
  GPU_f = AverageTime(attmpts);
  attmpts.clear();

  cout << endl;

  bool OMP_correct = CompareArrays<float>(resSEQ, resOMP, n);
  bool GPU_correct = CompareArrays<float>(resSEQ, resGPU, n);

  if (OMP_correct != true || GPU_correct != true) throw "Floats are different!";

  // DAXPY //

  n = threads_num;
  double CPU_d, OMP_d, GPU_d;

  double* x_d = CreateArray<double>(n);
  double* y_d = CreateArray<double>(n);

  double *res_d, *resOMP_d, *resGPU_d;

  // CPU //
  // SEQ //
  for (int i = 0; i < iter; i++) {
    start = omp_get_wtime();
    res_d = daxpy_seq(n, a, x_d, incx, y_d, incy);
    end = omp_get_wtime();
    attmpts.push_back(end - start);
  }
  CPU_d = AverageTime(attmpts);

  x_d = CreateArray<double>(n);
  y_d = CreateArray<double>(n);
  attmpts.clear();

  // OMP //
  for (int i = 0; i < iter; i++) {
    start = omp_get_wtime();
    resOMP_d = daxpy_omp(n, a, x_d, incx, y_d, incy);
    end = omp_get_wtime();
    attmpts.push_back(end - start);
  }
  OMP_d = AverageTime(attmpts);

  x_d = CreateArray<double>(n);
  y_d = CreateArray<double>(n);
  attmpts.clear();

  // GPU //
  for (int i = 0; i < iter; i++) {
    resGPU_d = daxpy_gpu(threads_num, block_size, n, a, x_d, incx, y_d, incy,
                         &attmpts);
  }

  GPU_d = AverageTime(attmpts);

  OMP_correct = CompareArrays<double>(res_d, resOMP_d, n);
  GPU_correct = CompareArrays<double>(res_d, resGPU_d, n);

  if (OMP_correct != true && GPU_correct != true)
    throw "Doubles are different!";

  PrintResults(n, CPU_f, OMP_f, GPU_f, CPU_d, OMP_d, GPU_d);

  delete[] x, y, resSEQ, resSEQ, res_d, resOMP, resOMP_d, resGPU, resGPU_d;

  return 0;
}