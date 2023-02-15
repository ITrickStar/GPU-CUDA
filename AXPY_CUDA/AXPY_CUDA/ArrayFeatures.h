#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <iostream>
#include <random>
#include <vector>

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

inline double AverageTime(std::vector<double> arr) {
  double res = 0;

  for (int i = 0; i < arr.size(); i++) res += arr[i];

  return res / arr.size();
}

inline int FindMinElementIdx(const std::vector<double> vec) {
  double min = vec[0];
  int idx = -1;

  for (int i = 1; i < vec.size(); i++) {
    if (vec[i] < min) {
      min = vec[i];
      idx = i;
    }
  }

  return idx;
}

template <typename T>
void PrintArray(T* arr, int size) {
  printf("\n");
  for (int i = 0; i < size; i++) std::cout << arr[i] << "; ";

  printf("\n");
}

template <typename T>
T* CreateArray(int size) {
  T* arr = new T[size];

  for (int i = 0; i < size; i++) arr[i] = (double)i * i / (i % 5);

  return arr;
}

template <typename T>
bool CompareArrays(T* a, T* b, int size) {
  for (int i = 0; i < size; i++) {
    if (std::fabs(a[i] - b[i]) > 0.000001) return false;
  }

  return true;
}
