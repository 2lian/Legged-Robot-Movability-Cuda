#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "collision.cu.h"
#include "cuda_runtime_api.h"
#include "cuda_util.h"
#include "one_leg.cu.h"
#include "thrust/copy.h"
#include "unified_math_cuda.cu.h"
#include "vector_functions.hpp"
#include <cstdio>
#include <iostream>
#include <limits>
#include <thrust/detail/copy.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/partition.h>
#include <thrust/transform.h>
#include <tuple>

__device__ inline void rotateInPlace(float3& point, float z_rot,
                                     float& cos_memory, float& sin_memory) {
    sincosf(z_rot, &sin_memory, &cos_memory);
    float buffer = point.x * sin_memory;
    point.x = point.x * cos_memory - point.y * sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

__device__ inline void unrotateInPlace(float3& point, float z_rot,
                                       float& cos_memory, float& sin_memory) {
    float buffer = point.x * -sin_memory;
    point.x = point.x * cos_memory - point.y * -sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

__device__ bool reachable_rotate_leg(float3 target, const float3 body_pos,
                                     const LegDimensions& dim) {
    {
        float cos_memory;
        float sin_memory;
        target.x -= body_pos.x;
        target.y -= body_pos.y;
        target.z -= body_pos.z;
        rotateInPlace(target, -dim.body_angle, cos_memory, sin_memory);
    }
    return reachability_absolute_tibia_limit(target, dim);
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
        // if (output.elements[body_index] == -1) {
        //     return;
        // }
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

        for (int leg = 1; leg < number_of_legs; ++leg) {
            min_value = min(min_value, arrays[leg].elements[i]);
            // min_value = min_value + arrays[leg].elements[i];
        }

        output.elements[i] = min_value;
    }
}

__global__ void find_min_kernel(thrust::device_vector<int>* arrays,
                                int number_of_legs,
                                thrust::device_vector<int> output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < output.size(); i += stride) {
        // for (int i = index; i < 1; i += stride) {
        int min_value = arrays[0][i];

        for (int leg = 1; leg < number_of_legs; ++leg) {
            // min_value = min(min_value, arrays[leg][i]);
            min_value = min_value + arrays[leg][i];
        }
        output[i] = min_value;
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
    int blockSize = 1024 / 1;
    int numBlock =
        (body_map.length * target_map.length + blockSize - 1) / blockSize;

    for (int leg_num = 0; leg_num < legs.length; leg_num++) {
        res_bool_array[leg_num].length = body_map.length;
        cudaMalloc(&(res_bool_array[leg_num].elements),
                   sizeof(int) * res_bool_array[leg_num].length);
        if (leg_num == 0) {
            cudaMemset(res_bool_array[leg_num].elements, 0,
                       sizeof(int) * res_bool_array[leg_num].length);
            float radius = legs.elements[0].body;
            float plus_z = 120;
            float minus_z = -60;
            in_cylinder_accu_kernel<<<numBlock, blockSize>>>(
                body_map, target_map, res_bool_array[leg_num], radius, plus_z,
                minus_z);
        } else {
            cudaMemcpy(res_bool_array[leg_num].elements,
                       res_bool_array[0].elements,
                       sizeof(int) * res_bool_array[leg_num].length,
                       cudaMemcpyDeviceToDevice);
        }
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("Cylinder and alloc");
    blockSize = 1024 / 2;
    numBlock =
        (body_map.length * target_map.length + blockSize - 1) / blockSize;
    for (int leg_num = 0; leg_num < legs.length; leg_num++) {
        CUDA_CHECK_ERROR("cudaKernel leg");
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

    blockSize = 1024;
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

std::tuple<Array<float3>, Array<int>>
robot_full_cccl(Array<float3> body_map, Array<float3> target_map,
                Array<LegDimensions> legs) {
    thrust::host_vector<float3> Body(body_map.elements,
                                     body_map.elements + body_map.length);
    thrust::host_vector<float3> Target(target_map.elements,
                                       target_map.elements + target_map.length);

    thrust::device_vector<float3> Body_g = Body;
    thrust::device_vector<float3> Target_g = Target;

    std::cout << Body_g.size() << std::endl;

    thrust::device_vector<float3>::iterator newEndBody = Body_g.end();
    thrust::device_vector<float3>::iterator newBeginBody = Body_g.begin();
    // newBeginBody += (newEndBody - newBeginBody) / 2;

    // Quaternion quat = quatFromVectAngle(make_float3(0, 0, 1), pI/2);
    Quaternion quat = quatFromVectAngle(make_float3(0, 1, 0), 0.4);
    auto my_func = QuaternionFunctor(quat);
    std::cout << quat.x << " | " << quat.y << " | " << quat.z << " | " << quat.w
              << " | " << std::endl;

    thrust::transform(newBeginBody, newEndBody, newBeginBody, my_func);
    thrust::transform(Target_g.begin(), Target_g.end(), Target_g.begin(),
                      my_func);

    thrust::device_vector<int>* result_ar;
    result_ar = new thrust::device_vector<int>[legs.length];
    size_t blockSize = 1024 / 1;
    size_t numBlock =
        ((newEndBody - newBeginBody) * Target_g.size() + blockSize - 1) /
        blockSize;
    thrust::device_vector<int>::iterator newEndResult;
    // int numBlock = (Body_g.size() + blockSize - 1) / blockSize;
    {
        auto dim = legs.elements[0];
        float sinr = sin(dim.coxa_pitch);
        float cosr = cos(dim.coxa_pitch);
        float radius = dim.body + cosr * dim.coxa_length + dim.femur_length +
                       dim.tibia_length;
        float plus_z =
            sinr * dim.coxa_length + dim.femur_length + dim.tibia_length;
        float minus_z =
            sinr * dim.coxa_length - dim.femur_length - dim.tibia_length;
        thrust::device_vector<int> not_far(newEndBody - newBeginBody);
        thrust::fill(not_far.begin(), not_far.end(), 0);
        cudaDeviceSynchronize();

        // in_cylinder_rec<<<1, 1>>>( // -1 if close enough
        auto ptr = thrust::raw_pointer_cast(Body_g.data()) +
                   (newBeginBody - Body_g.begin());
        auto s = newEndBody - newBeginBody;
        in_cylinder_cccl_kernel<<<numBlock, blockSize>>>( // -1 if close enough
            ptr, s, thrust::raw_pointer_cast(Target_g.data()), Target_g.size(),
            thrust::raw_pointer_cast(not_far.data()), radius, plus_z, minus_z);
        cudaDeviceSynchronize();

        newEndBody =
            thrust::partition(newBeginBody, newEndBody, not_far.begin(),
                              [] __device__(int x) { return x != 0; });
        // Body_g.erase(newEndBody, Body_g.end());
        std::cout << "Part 1 " << (newEndBody - newBeginBody) << std::endl;
    }
    CUDA_CHECK_ERROR("first cylinder and alloc");

    for (int leg_num = 0; leg_num < legs.length; leg_num++) {
        thrust::device_vector<int>& this_leg_result = result_ar[leg_num];
        this_leg_result.resize(newEndBody - newBeginBody);
        if (leg_num == 0) {
            thrust::fill(this_leg_result.begin(), this_leg_result.end(), 0);
            float radius = legs.elements[0].body;
            float plus_z = 120;
            float minus_z = -60;
            auto ptr = thrust::raw_pointer_cast(Body_g.data()) +
                       (newBeginBody - Body_g.begin());
            auto s = newEndBody - newBeginBody;
            in_cylinder_cccl_kernel<<<numBlock, blockSize>>>(
                ptr, s, thrust::raw_pointer_cast(Target_g.data()),
                Target_g.size(),
                thrust::raw_pointer_cast(this_leg_result.data()), radius,
                plus_z, minus_z);
            cudaDeviceSynchronize();

            newEndBody = thrust::partition(
                newBeginBody, newEndBody, this_leg_result.begin(),
                [] __device__(int x) { return x != -1; });
            // Body_g.erase(newEndBody, Body_g.end());
            std::cout << "Part 2 " << (newEndBody - newBeginBody) << std::endl;
            newEndResult = thrust::remove(this_leg_result.begin(),
                                          this_leg_result.end(), -1);
            this_leg_result.erase(newEndResult, this_leg_result.end());

        } else {
            thrust::copy(result_ar[0].begin(), result_ar[0].end(),
                         this_leg_result.begin());
        }
    }
    CUDA_CHECK_ERROR("Cylinder and alloc");
    blockSize = 1024 / 2;
    numBlock =
        ((newEndBody - newBeginBody) * Target_g.size() + blockSize - 1) /
        blockSize;

    for (int leg_num = 0; leg_num < legs.length; leg_num++) {
        CUDA_CHECK_ERROR("cudaKernel leg");
        auto ptr = thrust::raw_pointer_cast(Body_g.data()) +
                   (newBeginBody - Body_g.begin());
        const size_t s = (newEndBody - newBeginBody);
        Array<float3> b = {s, ptr};
        Array<float3> t = {Target_g.size(),
                           thrust::raw_pointer_cast(Target_g.data())};
        Array<int> r = {result_ar[leg_num].size(),
                        thrust::raw_pointer_cast(result_ar[leg_num].data())};
        reachable_leg_kernel_accu<<<numBlock, blockSize>>>(
            b, t, legs.elements[leg_num], r);
    }
    cudaDeviceSynchronize();

    CUDA_CHECK_ERROR("cudaMalloc final_count");

    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          result_ar[0].begin(), result_ar[1].begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          result_ar[0].end(), result_ar[1].end())),
                      result_ar[0].begin(), MinRowElement<int>());
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          result_ar[2].begin(), result_ar[3].begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          result_ar[2].end(), result_ar[3].end())),
                      result_ar[2].begin(), MinRowElement<int>());
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(
                          result_ar[0].begin(), result_ar[2].begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(
                          result_ar[0].end(), result_ar[2].end())),
                      result_ar[0].begin(), MinRowElement<int>());
    thrust::device_vector<int> final_count = result_ar[0];
    // blockSize = 1024;
    // numBlock = ((newEndBody - newBeginBody) + blockSize - 1) / blockSize;
    // find_min_kernel<<<numBlock, blockSize>>>(result_ar_gpu, legs.length,
    //                                          final_count);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("execution find_min_kernel");

    newEndBody =
        thrust::partition(newBeginBody, newEndBody, final_count.begin(),
                          [] __device__(int x) { return x != 0; });
    // Body_g.erase(newEndBody, Body_g.end());
    std::cout << (newEndBody - newBeginBody) << std::endl;
    newEndResult = thrust::remove(final_count.begin(), final_count.end(), 0);
    final_count.erase(newEndResult, final_count.end());

    auto my_UnFunc = UnQuaternionFunctor(quat);
    thrust::transform(newBeginBody, newEndBody, newBeginBody, my_UnFunc);
    thrust::transform(Target_g.begin(), Target_g.end(), Target_g.begin(),
                      my_UnFunc);

    thrust::host_vector<float3> oncpub(newEndBody - newBeginBody);
    thrust::copy(newBeginBody, newEndBody, oncpub.begin());
    Array<float3> out_body = thustVectToArray(oncpub);

    thrust::host_vector<int> oncpuc = final_count;
    Array<int> out_count = thustVectToArray(oncpuc);

    CUDA_CHECK_ERROR("cudaFree");
    auto out = std::make_tuple(out_body, out_count);
    std::cout << "cuda done" << std::endl;
    return out;
}
