#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "cuda_runtime_api.h"
#include "one_leg.cu.h"
#include "vector_types.h"
#include <cstdio>

__device__ void rotateInPlace(float3& point, float z_rot, float& cos_memory,
                              float& sin_memory) {
    sincosf(z_rot, &sin_memory, &cos_memory);
    float buffer = point.x * sin_memory;
    point.x = point.x * cos_memory - point.y * sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

__device__ void unrotateInPlace(float3& point, float z_rot, float& cos_memory,
                                float& sin_memory) {
    float buffer = point.x * -sin_memory;
    point.x = point.x * cos_memory - point.y * -sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

__device__ bool reachable_rotate_leg(float3 target, const float3 body_pos,
                                     const LegDimensions& dim) {
    float cos_memory;
    float sin_memory;
    target.x -= body_pos.x;
    target.y -= body_pos.y;
    target.z -= body_pos.z;
    rotateInPlace(target, -dim.body_angle, cos_memory, sin_memory);
    return reachability_vect(target, dim);
};

__global__ void reachable_leg_kernel_accu(Array<float3> body_map,
                                          Array<float3> target_map,
                                          LegDimensions dim,
                                          Array<int> output) {
    long index = blockIdx.x * blockDim.x + threadIdx.x;
    long stride = blockDim.x * gridDim.x;
    long maxid = (long)body_map.length * (long)target_map.length;
    for (long i = index; i < maxid; i += stride) {
        int body_index = i / target_map.length;
        int target_index = i % target_map.length;
        float3& target = target_map.elements[target_index];
        float3& body_pos = body_map.elements[body_index];
        if (reachable_rotate_leg(target, body_pos, dim)) {
            atomicAdd(&output.elements[body_index], 1);
        }
    }
};

__global__ void find_min_kernel(Array<int>* arrays, int number_of_legs,
                                Array<int> output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < output.length; i += stride) {
        // for (int i = index; i < 1; i += stride) {
        int min_value = arrays[0].elements[i];

        for (int j = 1; j < number_of_legs; ++j) {
            min_value = min(min_value, arrays[j].elements[i]);
            // min_value = min_value + arrays[j].elements[i];
        }

        output.elements[i] = min_value;
    }
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

Array<int> robot_full_reachable(Array<float3> body_map,
                                Array<float3> target_map,
                                Array<LegDimensions> legs) {
    {
        float3* newpointer;
        cudaMalloc(&newpointer, body_map.length * sizeof(float3));
        CUDA_CHECK_ERROR("cudaMalloc body_map");
        cudaMemcpy(newpointer, body_map.elements,
                   body_map.length * sizeof(float3), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR("cudaMemcpy body_map");
        body_map.elements = newpointer;
    }
    {
        float3* newpointer;
        cudaMalloc(&newpointer, target_map.length * sizeof(float3));
        CUDA_CHECK_ERROR("cudaMalloc target_map");
        cudaMemcpy(newpointer, target_map.elements,
                   target_map.length * sizeof(float3), cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR("cudaMemcpy target_map");
        target_map.elements = newpointer;
    }
    CUDA_CHECK_ERROR("cudaMalloc before leg");

    Array<int>* res_bool_array;
    res_bool_array = new Array<int>[legs.length];
    int blockSize = 1024;
    int numBlock =
        (body_map.length * target_map.length + blockSize - 1) / blockSize;

    for (int leg_num = 0; leg_num < legs.length; leg_num++) {
        res_bool_array[leg_num].length = body_map.length;
        cudaMalloc(&(res_bool_array[leg_num].elements),
                   sizeof(int) * res_bool_array[leg_num].length);
        CUDA_CHECK_ERROR("cudaMalloc leg");
        reachable_leg_kernel_accu<<<numBlock, blockSize>>>(
            body_map, target_map, legs.elements[leg_num],
            res_bool_array[leg_num]);
    }
    cudaDeviceSynchronize();

    CUDA_CHECK_ERROR("cudaMalloc reachable kernel");

    Array<int> final_count;
    final_count.length = res_bool_array[0].length;

    cudaMalloc(&final_count.elements, sizeof(int) * final_count.length);
    cudaMemset(final_count.elements, 0, final_count.length * sizeof(int));
    CUDA_CHECK_ERROR("cudaMalloc final_count");

    {
        Array<int>* newpointer;
        cudaMalloc(&newpointer, legs.length * sizeof(Array<int>));
        CUDA_CHECK_ERROR("cudaMalloc legdim");
        cudaMemcpy(newpointer, res_bool_array, legs.length * sizeof(Array<int>),
                   cudaMemcpyHostToDevice);
        CUDA_CHECK_ERROR("cudaMemcpy legdim");
        res_bool_array = newpointer;
    }

    numBlock = (body_map.length + blockSize - 1) / blockSize;
    find_min_kernel<<<numBlock, blockSize>>>(res_bool_array, legs.length,
                                             final_count);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("execution find_min_kernel");
    int* newpointer = new int[final_count.length];
    {
        cudaMemcpy(newpointer, final_count.elements,
                   final_count.length * sizeof(int), cudaMemcpyDeviceToHost);
        CUDA_CHECK_ERROR("cudaMemcpy final_count");
        cudaFree(final_count.elements);
        final_count.elements = newpointer;
    }

    cudaFree(body_map.elements);
    cudaFree(target_map.elements);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("cudaFree");
    return final_count;
}
