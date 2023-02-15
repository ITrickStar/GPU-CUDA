#include "CPU.h"

float* SEQMulti(float* A, float* B, int M, int N, int K) {
  float* C = MatrixZeros(M, K);
  TMatrix(B, N, K);

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      for (int l = 0; l < N; l++) {
        C[i * K + j] += A[i * N + l] * B[j * N + l];
      }
    }
  }

  TMatrix(B, K, N);
  return C;
}

float* OMPMulti(float* A, float* B, int M, int N, int K) {
  float* C = MatrixZeros(M, K);
  TMatrix(B, N, K);

#pragma omp parallel for schedule(static)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < K; j++) {
      for (int l = 0; l < N; l++) {
        C[i * K + j] += A[i * N + l] * B[j * N + l];
      }
    }
  }

  TMatrix(B, K, N);
  return C;
}
