#include "Utils.h"

double AverageTime(std::vector<double> arr) {
  double res = 0;

  for (int i = 0; i < arr.size(); i++) res += arr[i];

  return res / arr.size();
}

float* MatrixZeros(int M, int N) {
  int size = M * N;
  float* A = new float[size];
  for (int i = 0; i < size; i++) A[i] = 0;

  return A;
}

float* Matrix(int M, int N) {
  int size = M * N;
  float* A = new float[size];

  for (int i = 0; i < size; i++) A[i] = (float)((i * i) / 5);

  return A;
}

void PrintMatrix(float* A, int M, int N) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      printf("%-6.2f ", A[i * N + j]);
    }
    printf("\n");
  }

  printf("\n");
}

bool CompareMatrix(float* A, float* B, int M, int K) {
  for (int i = 0; i < M * K; i++)
    if (std::fabs(A[i] - B[i]) > 0.000001) return false;

  return true;
}

void TMatrix(float* A, int M, int N) {
  float* C = new float[M * N];

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      C[j * M + i] = A[i * N + j];
    }
  }
  for (int i = 0; i < M * N; i++) {
    A[i] = C[i];
  }

  delete[] C;
}
