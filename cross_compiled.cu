#include "cross_compiled.cuh"
#include "unified_math_cuda.cu.h"
#include "one_leg.cu.h"
#define BLOCSIZE 1024 / 4

#define CUDA_CHECK_ERROR(errorMessage)                                                   \
    do {                                                                                 \
        cudaError_t err = cudaGetLastError();                                            \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error in %s: %s\n", errorMessage,                      \
                    cudaGetErrorString(err));                                            \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

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
template <typename T_in, typename param, typename T_out>
float apply_kernel(const Array<T_in> input, const param dim,
                   void (*kernel)(const Array<T_in>, const param, Array<T_out> const),
                   Array<T_out> const output) {
    Array<T_in> gpu_in{};
    Array<T_out> gpu_out{};
    gpu_in.length = input.length;
    gpu_out.length = output.length;
    cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(T_in));
    CUDA_CHECK_ERROR("cudaMalloc gpu_in.elements");
    cudaMalloc(&gpu_out.elements, gpu_out.length * sizeof(T_out));
    CUDA_CHECK_ERROR("cudaMalloc gpu_out.elements");

    cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(T_in),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR("cudaMemcpy gpu_in.elements");

    constexpr int blockSize = BLOCSIZE;
    int numBlock = (input.length + blockSize - 1) / blockSize;
    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);
    // Do something on GPU
    kernel<<<numBlock, blockSize>>>(gpu_in, dim, gpu_out);
    // Stop event and sync
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK_ERROR("Kernel launch");

    cudaMemcpy(output.elements, gpu_out.elements, output.length * sizeof(T_out),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR("cudaMemcpy gpu_out.elements");
    cudaDeviceSynchronize();

    cudaFree(gpu_in.elements);
    cudaFree(gpu_out.elements);
    return elapsedTime;
}

template <typename T_in, typename param, typename T_out>
float apply_recurs(const Array<T_in> input, const param dim, Array<T_out> const output) {
    Array<T_in> gpu_in{};
    Array<T_out> gpu_out{};
    gpu_in.length = input.length;
    gpu_out.length = output.length;
    cudaMalloc(&gpu_in.elements, gpu_in.length * sizeof(T_in));
    CUDA_CHECK_ERROR("cudaMalloc gpu_in.elements");
    cudaMalloc(&gpu_out.elements, gpu_out.length * sizeof(T_out));
    CUDA_CHECK_ERROR("cudaMalloc gpu_out.elements");

    cudaMemcpy(gpu_in.elements, input.elements, gpu_in.length * sizeof(T_in),
               cudaMemcpyHostToDevice);
    CUDA_CHECK_ERROR("cudaMemcpy gpu_in.elements");

    constexpr int blockSize = BLOCSIZE;
    int numBlock = (input.length + blockSize - 1) / blockSize;
    Box box;
    box.center = make_float3(200, 0, 0);
    box.topOffset = make_float3(400, 1, 500);
    const uint max_quad_ind = pow(8, 3);
    numBlock = (max_quad_ind + blockSize - 1) / blockSize;
    // Prepare
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, 0);
    // Do something on GPU
    recursive_kernel<<<numBlock, blockSize>>>(box, gpu_in, dim, gpu_out, 0);
    // Stop event and sync
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
    // Clean up:
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    CUDA_CHECK_ERROR("Kernel launch");

    cudaMemcpy(output.elements, gpu_out.elements, output.length * sizeof(T_out),
               cudaMemcpyDeviceToHost);
    CUDA_CHECK_ERROR("cudaMemcpy gpu_out.elements");
    cudaDeviceSynchronize();

    cudaFree(gpu_in.elements);
    cudaFree(gpu_out.elements);
    return elapsedTime;
}
template float apply_recurs<float3, LegDimensions, float3>(Array<float3>,
                                                               LegDimensions,
                                                               Array<float3>);
// Explicit instantiation for float3, float3
template float apply_kernel<float3, LegDimensions, float3>(
    Array<float3>, LegDimensions, void (*)(Array<float3>, LegDimensions, Array<float3>),
    Array<float3>);

// Explicit instantiation for float3, bool
template float apply_kernel<float3, LegDimensions, bool>(
    Array<float3>, LegDimensions,
    void (*)(const Array<float3>, LegDimensions, Array<bool>), Array<bool>);

// Explicit instantiation for float3, float3
template float apply_kernel<float3, LegCompact, float3>(
    Array<float3>, LegCompact, void (*)(Array<float3>, LegCompact, Array<float3>),
    Array<float3>);

// Explicit instantiation for float3, bool
template float apply_kernel<float3, LegCompact, bool>(Array<float3>, LegCompact,
                                                      void (*)(const Array<float3>,
                                                               LegCompact, Array<bool>),
                                                      Array<bool>);
