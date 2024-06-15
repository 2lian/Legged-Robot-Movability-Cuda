#include "one_leg.cu"
#include "unified_math_cuda.cu.h"

__device__ __forceinline__ void z_rotateInPlace(float3& point, float z_rot,
                                                float& cos_memory, float& sin_memory) {
    sincosf(z_rot, &sin_memory, &cos_memory);
    float buffer = point.x * sin_memory;
    point.x = point.x * cos_memory - point.y * sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

__device__ __forceinline__ void z_unrotateInPlace(float3& point, float& cos_memory,
                                                  float& sin_memory) {
    float buffer = point.x * -sin_memory;
    point.x = point.x * cos_memory - point.y * -sin_memory;
    point.y = buffer + point.y * cos_memory;
    return;
}

/**
 * @brief will change the tibia_absolute_pos/neg according to the orientation
 *
 * @param quat
 * @param leg
 * @return new legdim
 */
__forceinline__ __device__ LegDimensions rotate_leg_data(Quaternion quat,
                                                         LegDimensions leg) {
    Quaternion quatOfLegAzimut = quatFromVectAngle(make_float3(0, 0, 1), leg.body_angle);

    Quaternion result =
        qtMultiply(qtMultiply(quatOfLegAzimut, quat), qtInvert(quatOfLegAzimut));
    float3 rpy = rpyFromQuat(result);
    float pitch = rpy.y;

    leg.tibia_absolute_pos -= pitch;
    leg.tibia_absolute_neg -= pitch;
    return leg;
}

__forceinline__ __device__ float3 make_asif_leg0(float3 point, LegDimensions leg,
                                                 float& cos_memory, float& sin_memory) {
    z_rotateInPlace(point, -leg.body_angle, cos_memory, sin_memory);
    return point;
}
__forceinline__ __device__ float3 undo_asif_leg0(float3 point, float cos_memory,
                                                 float sin_memory) {
    z_unrotateInPlace(point, cos_memory, sin_memory);
    return point;
}

template <typename Tout = bool, // function for reachability
          Tout (*reach_function)(const float3&,
                                 const LegDimensions&) = reachability_circles>
__device__ __forceinline__ Tout reachability_global(const float3& point,
                                                    const LegDimensions& dim,
                                                    const Quaternion quat) {
    __shared__ LegDimensions oriented_leg_dim;
    if (threadIdx.x == 0) {
        oriented_leg_dim = rotate_leg_data(quat, dim);
    }
    auto unrotated_point = qtInvRotate(quat, point);
    float cos_memory;
    float sin_memory;
    auto point_as_leg0 =
        make_asif_leg0(unrotated_point, oriented_leg_dim, cos_memory, sin_memory);
    Tout result = reach_function(point_as_leg0, oriented_leg_dim);
    return result;
}

__global__ void reachability_global_kernel(const Array<float3> input,
                                           const LegDimensions dim, Array<bool> output) {
    // __shared__ LegDimensions chached_dim;
    // if (threadIdx.x == 0) {
    //     chached_dim = dim;
    // }
    // __syncthreads();
    constexpr Quaternion quat = {1, 0, 0, 0};
    // constexpr Quaternion quat = {0.996, 0, 0.087, 0};
    // constexpr Quaternion quat = {0.985, 0, 0.174, 0};
    // constexpr Quaternion quat = {0.924, 0, 0.384, 0}; // y rot 40deg
    // constexpr Quaternion quat = {0.940, 0, 0, 0.342}; // z rot 40deg
    // quat = qtMultiply(quat, quat);
    // quat = qtMultiply(quat, quat);

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = reachability_global(input.elements[i], dim, quat);
    }
}
