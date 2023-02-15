#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>

#include <iostream>
#include <random>

#define BLOCK_SIZE 16

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
};

void gpuAssert(cudaError_t, const char*, int, bool);
double AverageTime(std::vector<double>);
float* MatrixZeros(int, int);
float* Matrix(int, int);
void PrintMatrix(float*, int, int);
bool CompareMatrix(float*, float*, int, int);
void TMatrix(float*, int, int);
