#include "CPU.h"

#include <omp.h>

#include <algorithm>

float* saxpy_seq(int n, float a, float* x, int incx, float* y, int incy) {
  for (int i = 0; i < n; i++) {
    if (i * incx >= n || i * incy >= n) break;
    y[i * incy] += a * x[i * incx];
  }

  return y;
}

double* daxpy_seq(int n, double a, double* x, int incx, double* y, int incy) {
  for (int i = 0; i < n; i++) {
    if (i * incx >= n || i * incy >= n) break;
    y[i * incy] += a * x[i * incx];
  }

  return y;
}

float* saxpy_omp(int n, float a, float* x, int incx, float* y, int incy) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    if (i * incx >= n || i * incy >= n) break;
    y[i * incy] += a * x[i * incx];
  }

  return y;
}

double* daxpy_omp(int n, double a, double* x, int incx, double* y, int incy) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < n; i++) {
    if (i * incx >= n || i * incy >= n) break;
    y[i * incy] += a * x[i * incx];
  }

  return y;
}
