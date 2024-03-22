#include "HeaderCPP.h"
#include "HeaderCUDA.h"
#include "collision.cu.h"
#include "cuda_runtime_api.h"
#include "cuda_util.h"
#include "one_leg.cu.h"
#include "thrust/copy.h"
#include "thrust/remove.h"
#include "unified_math_cuda.cu.h"
#include <algorithm>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <limits>
#include <ostream>
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

__device__ inline bool reachable_rotate_leg(float3 target,
                                            const float3 body_pos,
                                            const LegDimensions& dim) {
    {
        float cos_memory;
        float sin_memory;
        target.x -= body_pos.x;
        target.y -= body_pos.y;
        target.z -= body_pos.z;
        rotateInPlace(target, -dim.body_angle, cos_memory, sin_memory);
    }
    // if (target.x < 1000) {return false;}
    return reachability_absolute_tibia_limit(target, dim);
};

__global__ void reachable_leg_kernel_accu(Array<float3> body_map,
                                          Array<float3> target_map,
                                          LegDimensions dim,
                                          Array<int> output) {
    __shared__ LegDimensions sdim;
    if (threadIdx.x == 0) {
        sdim = dim;
    }
    __syncthreads();

    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    size_t maxid = (size_t)body_map.length * (size_t)target_map.length;
    for (size_t i = index; i < maxid; i += stride) {
        size_t body_index = i / target_map.length;
        size_t target_index = i % target_map.length;
        float3& target = target_map.elements[target_index];
        float3& body_pos = body_map.elements[body_index];
        if (reachable_rotate_leg(target, body_pos, sdim)) {
            atomicAdd(&output.elements[body_index], 1);
        }
    }
};

__global__ void reach_mem_kernel(float3* body_centers, const size_t Nb,
                                 float3* targets, const size_t Nt, int* output,
                                 LegDimensions dim) {
    __shared__ LegDimensions sdim;
    __shared__ int local_output;
    __shared__ float3 body_pos;

    auto center_index = blockIdx.x;
    auto target_index = threadIdx.x;

    if (target_index == 0) {
        sdim = dim;
        local_output = 0;
        body_pos = body_centers[center_index];
    }
    __syncthreads();

    if ((center_index < Nb) and (target_index < Nt)) {
        const float3 target = targets[target_index];
        if (reachable_rotate_leg(target, body_pos, sdim)) {
            local_output = 1;
        }
    }

    __syncthreads();
    if ((target_index == 0) && (local_output == 1)) {
        output[center_index] = 1;
    }
};

void launch_opti_mem_reach_kernel(Array<float3> body_map,
                                  Array<float3> target_map, LegDimensions dim,
                                  Array<int> output) {
    size_t processed_index = 0;
    size_t max_block_size = 1024 / 2;

    auto Nc = body_map.length;
    auto body_centers = body_map.elements;
    auto Nt = target_map.length;
    auto targets = target_map.elements;

    size_t numBlock = Nc;
    while (processed_index < Nt) {
        size_t targets_left_to_process = Nt - processed_index;
        size_t blockSize = std::min(targets_left_to_process, max_block_size);
        float3* sub_target_ptr = targets + processed_index;
        size_t sub_target_size = blockSize;

        // std::cout << "in_sphere_mem_kernel " << numBlock << " " << blockSize
        // << std::endl;
        reach_mem_kernel<<<numBlock, blockSize>>>(
            body_centers, Nc, sub_target_ptr, sub_target_size, output.elements,
            dim);

        processed_index += blockSize;
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

// Array<int> robot_full_reachable(Array<float3> body_map,
//                                 Array<float3> target_map,
//                                 Array<LegDimensions> legs) {
//     {
//         float3* newpointer;
//         cudaMalloc(&newpointer, body_map.length * sizeof(float3));
//         CUDA_CHECK_ERROR("cudaMalloc body_map");
//         cudaMemcpy(newpointer, body_map.elements,
//                    body_map.length * sizeof(float3), cudaMemcpyHostToDevice);
//         CUDA_CHECK_ERROR("cudaMemcpy body_map");
//         body_map.elements = newpointer;
//     }
//     {
//         float3* newpointer;
//         cudaMalloc(&newpointer, target_map.length * sizeof(float3));
//         CUDA_CHECK_ERROR("cudaMalloc target_map");
//         cudaMemcpy(newpointer, target_map.elements,
//                    target_map.length * sizeof(float3),
//                    cudaMemcpyHostToDevice);
//         CUDA_CHECK_ERROR("cudaMemcpy target_map");
//         target_map.elements = newpointer;
//     }
//     CUDA_CHECK_ERROR("cudaMalloc before leg");
//
//     Array<int>* res_bool_array;
//     res_bool_array = new Array<int>[legs.length];
//     int blockSize = 1024 / 1;
//     int numBlock =
//         (body_map.length * target_map.length + blockSize - 1) / blockSize;
//
//     for (int leg_num = 0; leg_num < legs.length; leg_num++) {
//         res_bool_array[leg_num].length = body_map.length;
//         cudaMalloc(&(res_bool_array[leg_num].elements),
//                    sizeof(int) * res_bool_array[leg_num].length);
//         if (leg_num == 0) {
//             cudaMemset(res_bool_array[leg_num].elements, 0,
//                        sizeof(int) * res_bool_array[leg_num].length);
//             float radius = legs.elements[0].body;
//             float plus_z = 120;
//             float minus_z = -60;
//             in_cylinder_accu_kernel<<<numBlock, blockSize>>>(
//                 body_map, target_map, res_bool_array[leg_num], radius,
//                 plus_z, minus_z);
//         } else {
//             cudaMemcpy(res_bool_array[leg_num].elements,
//                        res_bool_array[0].elements,
//                        sizeof(int) * res_bool_array[leg_num].length,
//                        cudaMemcpyDeviceToDevice);
//         }
//     }
//     cudaDeviceSynchronize();
//     CUDA_CHECK_ERROR("Cylinder and alloc");
//     blockSize = 1024 / 2;
//     numBlock =
//         (body_map.length * target_map.length + blockSize - 1) / blockSize;
//     for (int leg_num = 0; leg_num < legs.length; leg_num++) {
//         CUDA_CHECK_ERROR("cudaKernel leg");
//         reachable_leg_kernel_accu<<<numBlock, blockSize>>>(
//             body_map, target_map, legs.elements[leg_num],
//             res_bool_array[leg_num]);
//     }
//     cudaDeviceSynchronize();
//
//     CUDA_CHECK_ERROR("cudaMalloc reachable kernel");
//
//     Array<int> final_count;
//     final_count.length = res_bool_array[0].length;
//
//     cudaMalloc(&final_count.elements, sizeof(int) * final_count.length);
//     cudaMemset(final_count.elements, 0, final_count.length * sizeof(int));
//     CUDA_CHECK_ERROR("cudaMalloc final_count");
//
//     {
//         Array<int>* newpointer;
//         cudaMalloc(&newpointer, legs.length * sizeof(Array<int>));
//         CUDA_CHECK_ERROR("cudaMalloc legdim");
//         cudaMemcpy(newpointer, res_bool_array, legs.length *
//         sizeof(Array<int>),
//                    cudaMemcpyHostToDevice);
//         CUDA_CHECK_ERROR("cudaMemcpy legdim");
//         res_bool_array = newpointer;
//     }
//
//     blockSize = 1024;
//     numBlock = (body_map.length + blockSize - 1) / blockSize;
//     find_min_kernel<<<numBlock, blockSize>>>(res_bool_array, legs.length,
//                                              final_count);
//     cudaDeviceSynchronize();
//     CUDA_CHECK_ERROR("execution find_min_kernel");
//     int* newpointer = new int[final_count.length];
//     {
//         cudaMemcpy(newpointer, final_count.elements,
//                    final_count.length * sizeof(int), cudaMemcpyDeviceToHost);
//         CUDA_CHECK_ERROR("cudaMemcpy final_count");
//         cudaFree(final_count.elements);
//         final_count.elements = newpointer;
//     }
//
//     cudaFree(body_map.elements);
//     cudaFree(target_map.elements);
//     cudaDeviceSynchronize();
//     CUDA_CHECK_ERROR("cudaFree");
//     return final_count;
// }

class multi_rot_estimator {
  public:
    thrust::device_vector<float3> bodyWorking;
    thrust::device_vector<float3> targetRotated;
    thrust::device_vector<int>* resultLegArray;
    thrust::device_vector<int>& finalCount;

    thrust::device_vector<float3> bodyGlobal;
    thrust::device_vector<float3> targetGlobal;
    thrust::device_vector<float3>::iterator endBodyView;
    thrust::device_vector<float3>::iterator beginBodyView;
    Array<LegDimensions> legs;
    Array<LegDimensions> legsWorking;

    multi_rot_estimator(thrust::device_vector<float3> body,
                        thrust::device_vector<float3> target,
                        Array<LegDimensions> legsArray)
        : resultLegArray(new thrust::device_vector<int>[4]),
          finalCount(resultLegArray[0]) {

        bodyGlobal.resize(body.size());
        thrust::copy(body.begin(), body.end(), bodyGlobal.begin());

        targetGlobal.resize(target.size());
        thrust::copy(target.begin(), target.end(), targetGlobal.begin());

        legs = legsArray;
        legsWorking.length = legs.length;
        legsWorking.elements = new LegDimensions[legsWorking.length];
        bodyWorking.resize(bodyGlobal.size());
        thrust::copy(bodyGlobal.begin(), bodyGlobal.end(), bodyWorking.begin());
        targetRotated.resize(targetGlobal.size());
        thrust::copy(targetGlobal.begin(), targetGlobal.end(),
                     targetRotated.begin());

        std::cout << "init: " << (bodyGlobal.end() - bodyGlobal.begin())
                  << std::endl;

        eliminateAlwaysColliding();
        resetWorkingData();
    }

    void resetWorkingData() {
        bodyWorking.resize(endBodyView - beginBodyView);
        thrust::copy(targetGlobal.begin(), targetGlobal.end(),
                     targetRotated.begin());
        // std::cout << "body W address " << &(bodyWorking) << std::endl;
        thrust::copy(beginBodyView, endBodyView, bodyWorking.begin());
        for (int i = 0; i < legs.length; i++) {
            legsWorking.elements[i] = legs.elements[i];
        }
        return;
    }

    ~multi_rot_estimator() {
        delete[] resultLegArray;
        delete[] legsWorking.elements;
    }

    void flipWorkingSide() {
        beginBodyView = endBodyView;
        endBodyView = bodyGlobal.end();
    }

    void rotateData(Quaternion quat) {
        QuaternionFunctor my_func = QuaternionFunctor(quat);
        // std::cout << quat.x << " | " << quat.y << " | " << quat.z << " | "
        // << quat.w << " | " << std::endl;

        thrust::transform(bodyWorking.begin(), bodyWorking.end(),
                          bodyWorking.begin(), my_func);
        thrust::transform(targetRotated.begin(), targetRotated.end(),
                          targetRotated.begin(), my_func);
        std::cout << "Rotation done" << std::endl;
    };

    void eliminateAlwaysColliding() {
        thrust::device_vector<unsigned char> result(bodyGlobal.size());
        thrust::fill(result.begin(), result.end(), 0);
        float radius = 60;

        auto ptr = thrust::raw_pointer_cast(bodyGlobal.data());
        auto sizeBody = bodyGlobal.size();

        // size_t blockSize = 1024 / 1;
        // size_t numBlock =
        // (sizeBody * targetRotated.size() + blockSize - 1) / blockSize;
        // in_sphere_cccl_kernel<<<numBlock, blockSize>>>(
        //     ptr, sizeBody, thrust::raw_pointer_cast(targetRotated.data()),
        //     targetRotated.size(), thrust::raw_pointer_cast(result.data()),
        //     radius);
        launch_optimized_mem_in_sphere(
            ptr, sizeBody, thrust::raw_pointer_cast(targetRotated.data()),
            targetRotated.size(), thrust::raw_pointer_cast(result.data()),
            radius);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("eliminateAlwaysColliding");

        // thrust::host_vector<unsigned char> p = result;
        // std::cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] <<
        // "\n"
        //           << p[0 + 4] << " " << p[1 + 4] << " " << p[2 + 4] << " "
        //           << p[3 + 4] << "\n"
        //           << p[0 + 8] << " " << p[1 + 8] << " " << p[2 + 8] << " "
        //           << p[3 + 8] << std::endl;
        //
        auto newEnd = thrust::remove_if(
            bodyGlobal.begin(), bodyGlobal.end(), result.begin(),
            [] __device__(unsigned char x) { return x != 0; });
        bodyGlobal.erase(newEnd, bodyGlobal.end());
        newEnd = thrust::remove_if(
            bodyWorking.begin(), bodyWorking.end(), result.begin(),
            [] __device__(unsigned char x) { return x != 0; });
        bodyWorking.erase(newEnd, bodyWorking.end());
        beginBodyView = bodyGlobal.begin();
        endBodyView = bodyGlobal.end();

        std::cout << "eliminateAlwaysColliding: "
                  << (endBodyView - beginBodyView) << std::endl;
        // raise(1);
    }

    void eliminateTooFar() {
        auto dim = legsWorking.elements[0];
        float s_coxa_pitch = sin(dim.coxa_pitch);
        float c_coxa_pitch = cos(dim.coxa_pitch);
        float radius = dim.body + c_coxa_pitch * dim.coxa_length +
                       dim.femur_length + dim.tibia_length;
        auto plus_with_abs_limit =
            dim.tibia_length * sin(dim.tibia_absolute_pos) +
            dim.femur_length * sin(std::min(pI / 2, dim.max_angle_femur));
        float plus_z = s_coxa_pitch * dim.coxa_length + plus_with_abs_limit;
        float minus_z = s_coxa_pitch * dim.coxa_length - dim.femur_length -
                        dim.tibia_length;
        thrust::device_vector<unsigned char> not_far(bodyWorking.size());
        thrust::fill(not_far.begin(), not_far.end(), 0);

        auto ptr = thrust::raw_pointer_cast(bodyWorking.data());
        auto sizeBody = bodyWorking.size();

        // size_t blockSize = 1024 / 1;
        // size_t numBlock =
        //     (sizeBody * targetRotated.size() + blockSize - 1) / blockSize;
        // in_cylinder_cccl_kernel<<<numBlock, blockSize>>>( // -1 if close
        // enough
        //     ptr, sizeBody, thrust::raw_pointer_cast(targetRotated.data()),
        //     targetRotated.size(), thrust::raw_pointer_cast(not_far.data()),
        //     radius, plus_z, minus_z);
        launch_optimized_mem_in_cylinder( // 1 if close enough
            ptr, sizeBody, thrust::raw_pointer_cast(targetRotated.data()),
            targetRotated.size(), thrust::raw_pointer_cast(not_far.data()),
            radius, plus_z, minus_z);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("eliminateTooFar");

        endBodyView = thrust::partition(
            beginBodyView, endBodyView, not_far.begin(),
            [] __device__(unsigned char x) { return x != 0; });
        auto newEnd = thrust::remove_if(
            bodyWorking.begin(), bodyWorking.end(), not_far.begin(),
            [] __device__(unsigned char x) { return x == 0; });
        bodyWorking.erase(newEnd, bodyWorking.end());
        std::cout << "eliminateTooFar: " << (endBodyView - beginBodyView)
                  << std::endl;
    }

    void eliminateBodyColliding() {
        thrust::device_vector<unsigned char> result(bodyWorking.size());
        thrust::fill(result.begin(), result.end(), 0);
        float radius = legsWorking.elements[0].body;
        float plus_z = 250;
        float minus_z = -60;

        auto ptr = thrust::raw_pointer_cast(bodyWorking.data());
        auto sizeBody = bodyWorking.size();

        // size_t blockSize = 1024 / 1;
        // size_t numBlock =
        //     (sizeBody * targetRotated.size() + blockSize - 1) / blockSize;
        // in_cylinder_cccl_kernel<<<numBlock, blockSize>>>(
        //     ptr, sizeBody, thrust::raw_pointer_cast(targetRotated.data()),
        //     targetRotated.size(), thrust::raw_pointer_cast(result.data()),
        // radius, plus_z, minus_z);
        launch_optimized_mem_in_cylinder( // 1 if close enough
            ptr, sizeBody, thrust::raw_pointer_cast(targetRotated.data()),
            targetRotated.size(), thrust::raw_pointer_cast(result.data()),
            radius, plus_z, minus_z);
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("eliminateBodyColliding");

        endBodyView = thrust::partition(
            beginBodyView, endBodyView, result.begin(),
            [] __device__(unsigned char x) { return x == 0; });
        auto newEnd = thrust::remove_if(
            bodyWorking.begin(), bodyWorking.end(), result.begin(),
            [] __device__(unsigned char x) { return x != 0; });
        bodyWorking.erase(newEnd, bodyWorking.end());

        std::cout << "eliminateBodyColliding: " << (endBodyView - beginBodyView)
                  << std::endl;
    }

    void computeIndividualLegReachability(size_t leg_num) {
        thrust::device_vector<int>& this_leg_result = resultLegArray[leg_num];
        this_leg_result.resize(bodyWorking.size());
        thrust::fill(this_leg_result.begin(), this_leg_result.end(), 0);

        auto ptr = thrust::raw_pointer_cast(bodyWorking.data());
        const size_t sizeBody = bodyWorking.size();

        size_t blockSize = 1024 / 2;
        size_t numBlock =
            (sizeBody * targetRotated.size() + blockSize - 1) / blockSize;
        // std::cout << "numBlock: " << numBlock << std::endl;
        // std::cout << "sizeBody: " << sizeBody << std::endl;

        Array<float3> b = {sizeBody, ptr};
        Array<float3> t = {targetRotated.size(),
                           thrust::raw_pointer_cast(targetRotated.data())};
        Array<int> r = {this_leg_result.size(),
                        thrust::raw_pointer_cast(this_leg_result.data())};
        // reachable_leg_kernel_accu<<<numBlock, blockSize>>>(
        //     b, t, legsWorking.elements[leg_num], r);
        launch_opti_mem_reach_kernel(b, t, legsWorking.elements[leg_num], r);
        CUDA_CHECK_ERROR("computeIndividualLegReachability");
    }

    void computeAllLegReachability() {
        for (int leg_num = 0; leg_num < legsWorking.length; leg_num++) {
            computeIndividualLegReachability(leg_num);
        }
        cudaDeviceSynchronize();
        CUDA_CHECK_ERROR("computeAllLegReachability");
    }

    void agregateReachability() {
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(
                resultLegArray[0].begin(), resultLegArray[1].begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                resultLegArray[0].end(), resultLegArray[1].end())),
            resultLegArray[0].begin(), MinRowElement<int>());
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(
                resultLegArray[2].begin(), resultLegArray[3].begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                resultLegArray[2].end(), resultLegArray[3].end())),
            resultLegArray[2].begin(), MinRowElement<int>());
        thrust::transform(
            thrust::make_zip_iterator(thrust::make_tuple(
                resultLegArray[0].begin(), resultLegArray[2].begin())),
            thrust::make_zip_iterator(thrust::make_tuple(
                resultLegArray[0].end(), resultLegArray[2].end())),
            resultLegArray[0].begin(), MinRowElement<int>());
    }
    void cleanAgregated() {
        endBodyView =
            thrust::partition(beginBodyView, endBodyView, finalCount.begin(),
                              [] __device__(int x) { return x != 0; });
        auto newEnd = thrust::remove_if(
            bodyWorking.begin(), bodyWorking.end(), finalCount.begin(),
            [] __device__(int x) { return x == 0; });
        bodyWorking.erase(newEnd, bodyWorking.end());
    }
    void eliminateUnreachable() {
        std::cout << "reachabling legs";
        computeAllLegReachability();
        std::cout << ", gathering";
        agregateReachability();
        std::cout << ", cleaning";
        cleanAgregated();
        std::cout << ", done: ";
        std::cout << (endBodyView - beginBodyView) << std::endl;
    }

    void rotateOneLegLimit(Quaternion quat, LegDimensions& leg) {
        Quaternion quatOfLegAzimut =
            quatFromVectAngle(make_float3(0, 0, 1), leg.body_angle);

        Quaternion result = qtMultiply(qtMultiply(quatOfLegAzimut, quat),
                                       qtInvert(quatOfLegAzimut));
        float3 rpy = rpyFromQuat(result);
        float pitch = rpy.y;

        leg.tibia_absolute_pos -= pitch;
        leg.tibia_absolute_neg -= pitch;
    }

    void rotateLegsLimits(Quaternion quat) {
        for (int i = 0; i < legsWorking.length; i++) {
            rotateOneLegLimit(quat, legsWorking.elements[i]);
        }
    }

    void runPipeline(Quaternion quat) {
        resetWorkingData();
        rotateData(quat);
        rotateLegsLimits(quat);
        eliminateTooFar();
        eliminateBodyColliding();
        eliminateUnreachable();
        flipWorkingSide();
    } //1.427 in 132s

    thrust::host_vector<float3> getShavedResult() {
        thrust::host_vector<float3> oncpub(beginBodyView - bodyGlobal.begin());
        thrust::copy(bodyGlobal.begin(), beginBodyView, oncpub.begin());
        return oncpub;
    }
};

std::tuple<Array<float3>, Array<int>>
robot_full_struct(Array<float3> body_map, Array<float3> target_map,
                  Array<LegDimensions> legs) {

    thrust::host_vector<float3> Body(body_map.elements,
                                     body_map.elements + body_map.length);
    thrust::host_vector<float3> Target(target_map.elements,
                                       target_map.elements + target_map.length);
    thrust::device_vector<float3> Body_g = Body;
    thrust::device_vector<float3> Target_g = Target;

    multi_rot_estimator estimator = multi_rot_estimator(Body_g, Target_g, legs);
    Quaternion quatInit = quatFromVectAngle(make_float3(0, 0, 1), 0);
    // estimator.runPipeline(quatInit);

    float rollMin = -pI / 4;
    float rollMax = pI / 4;
    // float rollMin = 0;
    // float rollMax = 0;
    int rollSample = 4;

    // float pitchMin = -pI / 4;
    float pitchMin = -pI / 4;
    float pitchMax = +pI / 4;
    // float pitchMax = 0;
    int pitchSample = 4;

    float yawMin = 0;
    float yawMax = pI / 2;
    // float yawMax = 0;
    int yawSample = 4;

    for (int rollN = 0; rollN < rollSample; rollN++) {
        float rollX = (float)rollN / (float)rollSample;
        float roll = rollMin + (rollMax - rollMin) * rollX;
        Quaternion quatRoll = quatFromVectAngle(make_float3(1, 0, 0), roll);
        quatRoll = qtMultiply(quatRoll, quatInit);

        for (int pitchN = 0; pitchN <= pitchSample; pitchN++) {
            float pitchX = (float)pitchN / (float)pitchSample;
            float pitch = pitchMin + (pitchMax - pitchMin) * pitchX;
            Quaternion quatPitch =
                quatFromVectAngle(make_float3(0, 1, 0), pitch);
            quatPitch = qtMultiply(quatPitch, quatRoll);

            for (int yawN = 0; yawN < yawSample; yawN++) {
                float yawX = (float)yawN / (float)yawSample;
                float yaw = yawMin + (yawMax - yawMin) * yawX;
                std::cout << "" << std::endl;
                std::cout << "roll: " << roll << " | pitch: " << pitch
                          << " | yaw: " << yaw << std::endl;
                Quaternion quatYaw =
                    quatFromVectAngle(make_float3(0, 0, 1), yaw);
                quatYaw = qtMultiply(quatYaw, quatPitch);
                estimator.runPipeline(quatYaw);
            }
        }
    }
    // quat = quatFromVectAngle(make_float3(0, 0, 1), 1 * pI / 8);
    // estimator.runPipeline(quat);
    // quat = quatFromVectAngle(make_float3(0, 0, 1), 2 * pI / 8);
    // estimator.runPipeline(quat);
    // quat = quatFromVectAngle(make_float3(0, 0, 1), 3 * pI / 8);
    // estimator.runPipeline(quat);

    // estimator.flipWorkingSide();
    thrust::host_vector<float3> outBody = estimator.getShavedResult();
    thrust::host_vector<int> outCount(outBody.size());
    thrust::fill(outCount.begin(), outCount.end(), 3);

    Array<float3> out_body = thustVectToArray(outBody);
    Array<int> out_count = thustVectToArray(outCount);
    auto out = std::make_tuple(out_body, out_count);
    std::cout << "cuda done" << std::endl;
    return out;
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
    Quaternion quat = quatFromVectAngle(make_float3(0, 0, 1), 0.0);
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
    numBlock = ((newEndBody - newBeginBody) * Target_g.size() + blockSize - 1) /
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
