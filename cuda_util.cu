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

/**
 * @brief applies kernel on provided array to provided array
 * 
 * @tparam T_in
 * @tparam T_out
 * @param input Input array of any type
 * @param dim dimension of the leg
 * @param kernel cuda kernel to be used
 * @param output result is stored here
 */
template <typename T_in, typename T_out>
void apply_kernel(const Array<T_in> input, const LegDimensions dim,
                  void (*kernel)(const Array<T_in>, const LegDimensions,
                                 Array<T_out> const),
                  Array<T_out> const output) {
    Array<T_in> gpu_in{};
    Array<T_out> gpu_out{};
    gpu_in.length = input.length;
    gpu_out.length = output.length;
    cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(T_in));
    cudaMalloc(&gpu_out.elements, gpu_out.length * sizeof(T_out));

    cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(T_in),
               cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = (input.length + blockSize - 1) / blockSize;
    kernel<<<numBlock, blockSize>>>(gpu_in, dim, gpu_out);
    cudaDeviceSynchronize();

    cudaMemcpy(output.elements, gpu_out.elements, output.length * sizeof(T_out),
               cudaMemcpyDeviceToHost);

    cudaFree(gpu_in.elements);
    cudaFree(gpu_out.elements);
}

// Explicit instantiation for float3, float3
template void apply_kernel<float3, float3>(
    Array<float3>, LegDimensions,
    void (*)(Array<float3>, LegDimensions, Array<float3>), Array<float3>);

// Explicit instantiation for float3, bool
template void apply_kernel<float3, bool>(Array<float3>, LegDimensions,
                                         void (*)(const Array<float3>,
                                                  LegDimensions, Array<bool>),
                                         Array<bool>);

void apply_kernel(const Arrayf3 input, const LegDimensions dim,
                  void (*kernel)(const Arrayf3, const LegDimensions,
                                 Arrayf3 const),
                  Arrayf3 const output) {
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
    cudaDeviceSynchronize();

    cudaMemcpy(output.elements, gpu_out.elements,
               output.length * sizeof(float3), cudaMemcpyDeviceToHost);

    cudaFree(gpu_in.elements);
    cudaFree(gpu_out.elements);
}

void apply_kernel(const Arrayf3 input, const LegDimensions dim,
                  void (*kernel)(const Arrayf3, const LegDimensions,
                                 Arrayb const),
                  Arrayb const output) {
    Arrayf3 gpu_in{};
    Arrayb gpu_out{};
    gpu_in.length = input.length;
    gpu_out.length = output.length;
    cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(float3));
    cudaMalloc(&gpu_out.elements, gpu_out.length * sizeof(bool));

    cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(float3),
               cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int numBlock = (input.length + blockSize - 1) / blockSize;
    kernel<<<numBlock, blockSize>>>(gpu_in, dim, gpu_out);
    cudaDeviceSynchronize();

    cudaMemcpy(output.elements, gpu_out.elements, output.length * sizeof(bool),
               cudaMemcpyDeviceToHost);

    cudaFree(gpu_in.elements);
    cudaFree(gpu_out.elements);
}
