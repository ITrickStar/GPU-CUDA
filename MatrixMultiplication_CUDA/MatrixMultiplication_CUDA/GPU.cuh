#pragma once
#include <omp.h>
#include <stdio.h>

#include <vector>

#include "Utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

float* GPUMultiNaive(float* A, float* B, int M, int N, int K,
                     std::vector<double>* attmpts = nullptr);
float* GPUMultiOptimized(float* A, float* B, int M, int N, int K,
                         std::vector<double>* attmpts = nullptr);
