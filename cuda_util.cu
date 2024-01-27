#include "HeaderCPP.h"
#include "HeaderCUDA.h"
// #include "cuda_runtime_api.h"

__global__ void empty_kernel() {}

__device__ float sumOfSquares3df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1] +
           vector[2] * vector[2];
}

__device__ float sumOfSquares2df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1];
}

__global__ void norm3df_kernel(Arrayf3 input, Arrayf output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = norm3df(input.elements[i].x, input.elements[i].y,
                                     input.elements[i].z);
    }
}

void apply_kernel(Arrayf3 input, LegDimensions dim,
                  void (*kernel)(Arrayf3, LegDimensions, Arrayf3),
                  Arrayf3 output) {
    Arrayf3 gpu_in{};
    Arrayf3 gpu_out{};
    gpu_in.length = input.length;
    gpu_out.length = output.length;
    cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(float3));
    cudaMalloc(&gpu_out.elements, gpu_out.length * sizeof(float3));

    cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(float3),
               cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = (input.length + blockSize - 1) / blockSize;
    kernel<<<numBlock, blockSize>>>(gpu_in, dim, gpu_out);

    cudaMemcpy(output.elements, gpu_out.elements,
               output.length * sizeof(float3), cudaMemcpyDeviceToHost);

   cudaFree(gpu_in.elements);
   cudaFree(gpu_out.elements);
}
