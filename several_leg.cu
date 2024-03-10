#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "collision.cu.h"
#include "one_leg.cu.h"
#include "thrust/detail/copy.h"
#include "thrust/detail/raw_pointer_cast.h"
#include "thrust/device_vector.h"
#include "thrust/fill.h"
#include "thrust/transform.h"
#include <cstdio>
#include <iostream>

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
        if (reachable_rotate_leg(target, body_pos, dim) or true) {
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

struct CylinderFunctor {
    float radius;
    float plus_z;
    float minus_z;

    __host__ __device__ CylinderFunctor(float _radius, float _plus_z,
                                        float _minus_z)
        : radius(_radius), plus_z(_plus_z), minus_z(_minus_z) {}

    __device__ int operator()(float3 body, float3 target) const {
        bool r = in_cylinder(radius, plus_z, minus_z, body, target);
        // bool r = in_cylinder(10, 10, 10, body, target);
        return r ? -1 : 0;
    }
};

Array<int> robot_full_cccl(Array<float3> body_map, Array<float3> target_map,
                           Array<LegDimensions> legs) {
    thrust::host_vector<float3> Body(body_map.elements,
                                     body_map.elements + body_map.length);
    thrust::host_vector<float3> Target(target_map.elements,
                                       target_map.elements + target_map.length);

    thrust::device_vector<float3> Body_g = Body;
    thrust::device_vector<float3> Target_g = Target;

    thrust::device_vector<int>* result_ar;

    result_ar = new thrust::device_vector<int>[legs.length];
    int blockSize = 1024 / 1;
    int numBlock =
        (Body_g.size() * Target_g.size() + blockSize - 1) / blockSize;

    for (int leg_num = 0; leg_num < legs.length; leg_num++) {
        thrust::device_vector<int>& this_leg_result = result_ar[leg_num];
        this_leg_result.resize(Body_g.size());
        if (leg_num == 0) {
            thrust::fill(this_leg_result.begin(), this_leg_result.end(), 0);
            float radius = legs.elements[0].body;
            float plus_z = 120;
            float minus_z = -60;
            CylinderFunctor cylinder_func(radius, plus_z, minus_z);
            in_cylinder_cccl_kernel<<<numBlock, blockSize>>>(
                thrust::raw_pointer_cast(Body_g.data()), Body_g.size(),
                thrust::raw_pointer_cast(Target_g.data()), Target_g.size(),
                thrust::raw_pointer_cast(this_leg_result.data()), radius,
                plus_z, minus_z);
            cudaDeviceSynchronize();

            thrust::device_vector<float3>::iterator newEndBody =
                thrust::remove_if(Body_g.begin(), Body_g.end(),
                                  this_leg_result.begin(),
                                  [] __device__(int x) { return x == -1; });
            Body_g.erase(newEndBody, Body_g.end());
            std::cout << Body_g.size() << std::endl;
            thrust::device_vector<int>::iterator newEndResult = thrust::remove(
                this_leg_result.begin(), this_leg_result.end(), -1);
            this_leg_result.erase(newEndResult, this_leg_result.end());
            std::cout << this_leg_result.size() << std::endl;

        } else {
            thrust::copy(result_ar[0].begin(), result_ar[0].end(),
                         this_leg_result.begin());
        }
    }
    CUDA_CHECK_ERROR("Cylinder and alloc");
    blockSize = 1024 / 2;
    numBlock = (Body_g.size() * Target_g.size() + blockSize - 1) / blockSize;
    for (int leg_num = 0; leg_num < legs.length; leg_num++) {
        CUDA_CHECK_ERROR("cudaKernel leg");
        Array<float3> b = {Body_g.size(),
                           thrust::raw_pointer_cast(Body_g.data())};
        Array<float3> t = {Target_g.size(),
                           thrust::raw_pointer_cast(Target_g.data())};
        Array<int> r = {result_ar[leg_num].size(),
                        thrust::raw_pointer_cast(result_ar[leg_num].data())};
        reachable_leg_kernel_accu<<<numBlock, blockSize>>>(
            b, t, legs.elements[leg_num], r);
    cudaDeviceSynchronize();
        int sum = thrust::reduce(result_ar[leg_num].begin(),
                                 result_ar[leg_num].end(), 0);
        std::cout << "Sum of the elements: " << sum << std::endl;
    }
    cudaDeviceSynchronize();

    CUDA_CHECK_ERROR("cudaMalloc reachable kernel");

    thrust::device_vector<int> final_count(result_ar[0].size());

    CUDA_CHECK_ERROR("cudaMalloc final_count");

    thrust::device_vector<int>* result_ar_gpu;
    cudaMalloc(&result_ar_gpu,
               legs.length * sizeof(thrust::device_vector<int>));
    CUDA_CHECK_ERROR("cudaMalloc legdim");
    cudaMemcpy(result_ar_gpu, result_ar,
               legs.length * sizeof(thrust::device_vector<int>),
               cudaMemcpyHostToDevice);
    // delete result_ar;
    CUDA_CHECK_ERROR("cudaMemcpy legdim");
    result_ar_gpu = result_ar_gpu;

    blockSize = 1024;
    numBlock = (Body_g.size() + blockSize - 1) / blockSize;
    find_min_kernel<<<numBlock, blockSize>>>(result_ar_gpu, legs.length,
                                             final_count);
    cudaDeviceSynchronize();
    CUDA_CHECK_ERROR("execution find_min_kernel");

    thrust::device_vector<float3>::iterator newEndBody =
        thrust::remove_if(Body_g.begin(), Body_g.end(), final_count.begin(),
                          [] __device__(int x) { return x == 0; });
    Body_g.erase(newEndBody, Body_g.end());
    std::cout << Body_g.size() << std::endl;
    thrust::device_vector<int>::iterator newEndResult =
        thrust::remove(final_count.begin(), final_count.end(), 0);
    final_count.erase(newEndResult, final_count.end());
    std::cout << final_count.size() << std::endl;

    Array<int> out;
    thrust::host_vector<int> oncpu = final_count;
    out.length = oncpu.size();
    out.elements = new int[out.length];
    std::copy(oncpu.data(), oncpu.data() + oncpu.size(), out.elements);

    CUDA_CHECK_ERROR("cudaFree");
    return out;
}
