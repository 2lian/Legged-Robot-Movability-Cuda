#pragma once
#include "HeaderCPP.h"
#include "HeaderCUDA.h"

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

template <typename Tred, typename Tcheck, class F>
__global__ void double_reduction_kernel(Tred* toReduce, size_t Nred,
                                        Tcheck* toCheck, size_t Ncheck,
                                        unsigned char* output,
                                        bool (*checkFunction)(Tred, Tcheck));

template <typename Tred, typename Tcheck, class F>
void launch_double_reduction(Tred* toReduce, const size_t Nred, Tcheck* toCheck,
                             const size_t Ncheck, unsigned char* output,
                             // bool (*checkFunction)(Tred, Tcheck));
                             F checkFunction);
