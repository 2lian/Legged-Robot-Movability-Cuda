#pragma once
#include "HeaderCPP.h"
#include "HeaderCUDA.h"
// #include <driver_types.h>
// #include <thrust/copy.h>
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <thrust/remove.h>
// #include <thrust/sequence.h>

__global__ void empty_kernel();
// __host__ float norm3df(float x, float y, float z);

__device__ __host__ inline float sumOfSquares3df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1] +
           vector[2] * vector[2];
}

__device__ __host__ inline float sumOfSquares2df(const float* vector) {
    return vector[0] * vector[0] + vector[1] * vector[1];
}

#define CUDA_CHECK_ERROR(errorMessage)                                         \
    do {                                                                       \
        cudaError_t err = cudaGetLastError();                                  \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s: %s\n", errorMessage,            \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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

template <typename T>
Array<T> thustVectToArray(thrust::device_vector<T> thrust_vect);

template <typename T>
Array<T> thustVectToArray(thrust::host_vector<T> thrust_vect);

template <typename T>
thrust::device_vector<T> arrayToThrustVect(Array<T> array);

template <typename T> thrust::host_vector<T> arrayToThrustVect(Array<T> array);

template <typename MyType>
struct MinRowElement
    : public thrust::unary_function<thrust::tuple<MyType, MyType>, MyType> {
    __device__ MyType operator()(const thrust::tuple<MyType, MyType>& t) const;
};

// template <typename Tred, typename Tcheck, class F>
// __global__ void double_reduction_kernel(Tred* toReduce, size_t Nred,
//                                         Tcheck* toCheck, size_t Ncheck,
//                                         unsigned char* output,
//                                         bool (*checkFunction)(Tred, Tcheck));
//
// template <typename Tred, typename Tcheck, class F>
// void launch_double_reduction(Tred* toReduce, const size_t Nred, Tcheck*
// toCheck,
//                              const size_t Ncheck, unsigned char* output,
//                              // bool (*checkFunction)(Tred, Tcheck));
//                              F checkFunction);

template <typename Tred, typename Tcheck, class F>
__global__ void
double_reduction_kernel(Tred* toReduce, int Nred, Tcheck* toCheck, int Ncheck,
                        unsigned char* output, F checkFunction) {
    __shared__ bool result;
    __shared__ float3 data;
    const auto& data_index = blockIdx.x;
    const auto& target_index = threadIdx.x;

    if (target_index == 0) {
        result = false;
        data = toReduce[data_index];
    }
    __syncthreads();
    if (not((data_index < Nred) and (target_index < Ncheck))) {
        return;
    }

    const float3 target = toCheck[target_index];
    bool r = checkFunction(data, target);
    int wrap_result = __any_sync(__activemask(), r);
    int laneId = threadIdx.x & 0x1f;
    if (laneId == 0 and wrap_result!=0) {
        result = true;
    }
    // if (r) {
    // result = true;
    // }

    __syncthreads();
    if ((target_index == 0) && (result)) {
        output[data_index] = 1;
    }
}

template <typename Tred, typename Tcheck, class F>
void launch_double_reduction(Tred* toReduce, const size_t Nred, Tcheck* toCheck,
                             const size_t Ncheck, unsigned char* output,
                             F checkFunction, cudaStream_t strm) {
    size_t processed_index = 0;
    size_t max_block_size = 1024 / 1;
    // size_t max_block_size = 1;
    size_t numBlock = Nred;
    while (processed_index < Ncheck) {
        size_t targets_left_to_process = Ncheck - processed_index;
        size_t blockSize = std::min(targets_left_to_process, max_block_size);
        Tred* sub_target_ptr = toCheck + processed_index;
        size_t sub_target_size = blockSize;

        double_reduction_kernel<<<numBlock, blockSize, 0, strm>>>(
            toReduce, Nred, sub_target_ptr, sub_target_size, output,
            checkFunction);

        processed_index += blockSize;
    }
}

template <typename Tred, typename Tcheck, class F>
void launch_double_reduction(Tred* toReduce, const size_t Nred, Tcheck* toCheck,
                             const size_t Ncheck, unsigned char* output,
                             F checkFunction) {
    size_t processed_index = 0;
    size_t max_block_size = 1024 / 1;
    // size_t max_block_size = 1;
    size_t numBlock = Nred;
    while (processed_index < Ncheck) {
        size_t targets_left_to_process = Ncheck - processed_index;
        size_t blockSize = std::min(targets_left_to_process, max_block_size);
        Tred* sub_target_ptr = toCheck + processed_index;
        size_t sub_target_size = blockSize;

        double_reduction_kernel<<<numBlock, blockSize>>>(
            toReduce, Nred, sub_target_ptr, sub_target_size, output,
            checkFunction);

        processed_index += blockSize;
    }
}

template <typename Tred, typename Tcheck, class F1, class F2>
__global__ void double_reduction_kernel(Tred* toReduce, const int Nred,
                                        Tcheck* toCheck, const int Ncheck,
                                        unsigned char* outputValidate,
                                        unsigned char* outputEliminate,
                                        F1 validateFunc, F2 eliminateFunc) {
    __shared__ bool result1;
    __shared__ bool result2;
    __shared__ float3 data;
    const auto& data_index = blockIdx.x;
    const auto& target_index = threadIdx.x;

    if (target_index == 0) {
        result1 = false;
        result2 = true;
        data = toReduce[data_index];
    }
    __syncthreads();
    if (not((data_index < Nred) and (target_index < Ncheck))) {
        return;
    }

    const float3 target = toCheck[target_index];
    int thrd_r1 = __any_sync(__activemask(), validateFunc(data, target));
    int thrd_r2 = __any_sync(__activemask(), eliminateFunc(data, target));
    int laneId = threadIdx.x & 0x1f;
    if (laneId == 0 and thrd_r1!=0) {
        result1 = true;
    }
    if (laneId == 0 and thrd_r2!=0) {
        result2 = false;
    }
    // if (validateFunc(data, target)) {
    //     result1 = true;
    // }
    // if (eliminateFunc(data, target)) {
    //     result2 = false;
    // }

    __syncthreads();
    if ((target_index == 0) && (result1)) {
        outputValidate[data_index] = 1;
    }
    if ((target_index == 0) && (not result2)) {
        outputEliminate[data_index] = 1;
    }
}

__global__ void bring_together_kernel(unsigned char* outValidate,
                                      unsigned char* outEliminate, size_t N);

template <typename Tred, typename Tcheck, class F1, class F2>
void launch_double_reduction(Tred* toReduce, const size_t Nred, Tcheck* toCheck,
                             const size_t Ncheck, unsigned char* output,
                             // bool (*checkFunction)(Tred, Tcheck)) {
                             F1 validateFunc, F2 eliminateFunc) {

    size_t processed_index = 0;
    size_t max_block_size = 1024 / 1;
    // size_t max_block_size = 1;
    size_t numBlock = Nred;
    auto& outputValidate = output;
    unsigned char* outputEliminate;
    cudaMalloc(&outputEliminate, Nred * sizeof(unsigned char));
    cudaMemset(outputEliminate, 0, Nred * sizeof(unsigned char));
    while (processed_index < Ncheck) {
        // cudaStream_t sub_stream;
        // cudaStreamCreate(&sub_stream);
        size_t targets_left_to_process = Ncheck - processed_index;
        size_t blockSize = std::min(targets_left_to_process, max_block_size);
        Tred* sub_target_ptr = toCheck + processed_index;
        size_t sub_target_size = blockSize;

        double_reduction_kernel<<<numBlock, blockSize>>>(
            toReduce, Nred, sub_target_ptr, sub_target_size, outputValidate,
            outputEliminate, validateFunc, eliminateFunc);

        processed_index += blockSize;
    }
    int blockSize = 1024;
    numBlock = (Nred + blockSize - 1) / blockSize;
    // cudaDeviceSynchronize();
    bring_together_kernel<<<numBlock, blockSize>>>(outputValidate,
                                                   outputEliminate, Nred);
    // cudaDeviceSynchronize();
}
