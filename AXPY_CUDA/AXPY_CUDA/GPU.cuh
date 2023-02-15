#include <vector>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

float* saxpy_gpu(int threads_num, int block_size, int n, float a, float* x,
                 int incx, float* y, int incy,
                 std::vector<double>* times = nullptr);

double* daxpy_gpu(int threads_num, int block_size, int n, double a, double* x,
                  int incx, double* y, int incy,
                  std::vector<double>* times = nullptr);
