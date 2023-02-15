#pragma once
#include <omp.h>

#include "Utils.h"

float* SEQMulti(float* A, float* B, int M, int N, int K);
float* OMPMulti(float* A, float* B, int M, int N, int K);
