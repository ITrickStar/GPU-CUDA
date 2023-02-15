#pragma once
#include <stdio.h>

float* saxpy_seq(int n, float a, float* x, int incx, float* y, int incy);
double* daxpy_seq(int n, double a, double* x, int incx, double* y, int incy);

float* saxpy_omp(int n, float a, float* x, int incx, float* y, int incy);
double* daxpy_omp(int n, double a, double* x, int incx, double* y, int incy);
