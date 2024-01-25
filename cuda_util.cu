#include "HeaderCUDA.h"

__global__ void empty_kernel() {}

__device__ float sumOfSquares3df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1] +
           vector[2] * vector[2];
}

__device__ float sumOfSquares2df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1];
}

__global__ void norm3df_kernel(Matrixf table, Matrixf result_table) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < table.height; i += stride) {
        result_table.elements[i] = norm3df(table.elements[i * table.width],
                                           table.elements[i * table.width + 1],
                                           table.elements[i * table.width + 2]);
    }
}

