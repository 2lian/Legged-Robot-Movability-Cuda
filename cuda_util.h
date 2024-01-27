#include "HeaderCPP.h"
#include "HeaderCUDA.h"
// #include "cuda_runtime_api.h"

__global__ void empty_kernel();
__device__ float sumOfSquares3df(const float* vector);
__device__ float sumOfSquares2df(const float* vector);
__global__ void norm3df_kernel(Arrayf3 input, Arrayf output);
void apply_kernel(Arrayf3 input, LegDimensions dim,
                  void (*kernel)(Arrayf3, LegDimensions, Arrayf3),
                  Arrayf3 output);
