#pragma once
#include "HeaderCPP.h"
#include "HeaderCUDA.h"
// #include "cuda_runtime_api.h"

__global__ void empty_kernel();
__device__ float sumOfSquares3df(const float* vector);
__device__ float sumOfSquares2df(const float* vector);
__global__ void norm3df_kernel(Arrayf3 input, Arrayf output);

/* void apply_kernel(const Arrayf3 input, const LegDimensions dim, */
/*                   void (*kernel)(const Arrayf3, const LegDimensions, Arrayf3
 * const), */
/*                   Arrayf3 const output); */
/* void apply_kernel(const Arrayf3 input, const LegDimensions dim, */
/*                   void (*kernel)(const Arrayf3, const LegDimensions, Arrayb
 * const), */
/*                   Arrayb const output); */

template <typename T_in, typename T_out>
void apply_kernel(const Array<T_in> input, const LegDimensions dim,
                  void (*kernel)(const Array<T_in>, const LegDimensions,
                                 Array<T_out> const),
                  Array<T_out> const output);
