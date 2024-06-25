#include "cross_compiled.cuh"
#include "one_leg.cu.h"
#include "settings.h"
#include "unified_math_cuda.cu.h"
#include <chrono>
#include <iostream>
#include <ostream>
#include <thread>
#define BLOCSIZE 1024 / 4
using namespace std::chrono_literals;

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
    kernel<<<numBlock, blockSize>>>(gpu_in, dim, gpu_out); // warmup
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
    box.center = make_float3(BoxCenter[0], BoxCenter[1], BoxCenter[2]);
    box.topOffset = make_float3(BoxSize[0], BoxSize[1], BoxSize[2]);
    const uint max_quad_ind = pow(8, 2);
    numBlock = (max_quad_ind + blockSize - 1) / blockSize;
    // recursive_kernel<<<1, 24>>>(box, gpu_in, dim, gpu_out, 30);
    // Prepare
    cudaEvent_t start, stop;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
    cudaEventRecord(start, stream);
    // Do something on GPU
    recursive_kernel<<<numBlock, blockSize, 0, stream>>>(box, gpu_in, dim, gpu_out, 0, 0,
                                                         false);
    // recursive_kernel<<<numBlock, blockSize>>>(box, gpu_in, dim, gpu_out, 0, 0, false);
    // recursive_kernel<<<1, 24>>>(box, gpu_in, dim, gpu_out, 0, 0);
    // Stop event and sync
    // cudaDeviceSynchronize();
    // std::this_thread::sleep_for(1s);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
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
template float apply_recurs<float3, LegDimensions, float3>(Array<float3>, LegDimensions,
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

__host__ double apply_dist_cpu(const Array<float3> input, const LegDimensions dim,
                           Array<float3> const output) {
    auto start = std::chrono::high_resolution_clock::now();
    distance_kernel_cpu(input, dim, output);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (end - start)*1000;
    return duration.count();
}

__host__ double apply_reach_cpu(const Array<float3> input, const LegDimensions dim,
                           Array<bool> const output) {
    auto start = std::chrono::high_resolution_clock::now();
    // std::cout << "start" << std::endl;
    reachability_kernel_cpu(input, dim, output);
    // std::cout << "end" << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = (end - start)*1000;
    return duration.count();
}
