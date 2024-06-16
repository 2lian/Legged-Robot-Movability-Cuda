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

template <typename Tout = bool, // function for distance
          Tout (*reach_function)(float3&, const LegDimensions&) = distance_circles>
__device__ __forceinline__ Tout distance_global(float3& point, const LegDimensions& dim,
                                                const Quaternion quat) {
    __shared__ LegDimensions oriented_leg_dim;
    if (threadIdx.x == 0) {
        oriented_leg_dim = rotate_leg_data(quat, dim);
    }
    auto unrotated_point = qtInvRotate(quat, point);
    float cos_memory;
    float sin_memory;
    __syncthreads();
    auto point_as_leg0 =
        make_asif_leg0(unrotated_point, oriented_leg_dim, cos_memory, sin_memory);
    Tout result = reach_function(point_as_leg0, oriented_leg_dim);
    auto rerotated_point = undo_asif_leg0(point_as_leg0, cos_memory, sin_memory);
    rerotated_point = qtRotate(quat, rerotated_point);
    point = rerotated_point;
    return result;
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
    __syncthreads();
    auto point_as_leg0 =
        make_asif_leg0(unrotated_point, oriented_leg_dim, cos_memory, sin_memory);
    Tout result = reach_function(point_as_leg0, oriented_leg_dim);
    return result;
}

constexpr Quaternion quat = {1, 0, 0, 0};
// constexpr Quaternion quat = {0.999, 0, 0.01, 0};
// constexpr Quaternion quat = {0.996, 0, -0.087, 0};
// constexpr Quaternion quat = {0.985, 0, 0.174, 0};
// constexpr Quaternion quat = {0.924, 0, -0.384, 0}; // y rot 40deg
// constexpr Quaternion quat = {0.940, 0, 0, 0.342}; // z rot 40deg
__global__ void reachability_global_kernel(const Array<float3> input,
                                           const LegDimensions dim, Array<bool> output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = reachability_global(input.elements[i], dim, quat);
    }
}
__global__ void distance_global_kernel(const Array<float3> input, const LegDimensions dim,
                                       Array<float3> output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        float3 result = input.elements[i];
        bool reachability = distance_global(result, dim, quat);
        output.elements[i] = result;
    }
}

__device__ __forceinline__ float3 centerPoint(float3 p1, float3 p2) {
    float3 center;
    center.x = p1.x + p2.x;
    center.y = p1.y + p2.y;
    center.z = p1.z + p2.z;
    return center;
}

template <typename Tin>
__host__ __device__ __forceinline__ bool readBit(Tin data, uchar bitPosition) {
    return (data >> bitPosition) & 1;
}

__device__ __forceinline__ float3 flipVectorOnQuad(float3 vect, uchar quadNumber) {
    if (readBit(quadNumber, 0)) {
        vect.x *= -1;
    }
    if (readBit(quadNumber, 1)) {
        vect.y *= -1;
    }
    if (readBit(quadNumber, 2)) {
        vect.z *= -1;
    }
    return vect;
}

__global__ void fillOutKernel(Box box, float3 distance, Array<float3> input,
                              Array<float3> output) {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    auto stride = blockDim.x * gridDim.x;
    for (auto i = index; i < input.length; i += stride) {
        auto in = input.elements[i];
        auto& out = output.elements[i];
        float3 delta = in - box.center;
        delta = abs(delta);
        auto boxEdge = abs(box.topOffset);
        bool inside = boxEdge.x > delta.x and boxEdge.y > delta.y and boxEdge.z > delta.z;
        if (inside) {
            out = make_float3(222, 222, 222);
            // out = distance;
        }
    }
}

#define MAX_DEPTH 1
__global__ void recursive_kernel(Box box, const Array<float3> input,
                                 const LegDimensions leg, Array<float3> output,
                                 uchar depth) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (uchar quadr = index; quadr < 8; quadr += stride) {
        auto center = box.center;
        float3 distance = center;
        bool reachability = distance_circles(distance, leg);
        distance = flipVectorOnQuad(distance, quadr);
        // float3 edge = flipVectorOnQuad(box.topOffset, quadr);
        float3 edge = box.topOffset;
        bool isInOrigin = distance.x > 0 and distance.y > 0 and distance.z > 0;
        bool isInSubBox =
            distance.x < edge.x and distance.y < edge.y and distance.z < edge.z;
        bool reachabilityEdgeInTheBox = isInOrigin && isInSubBox;
        Box new_box;
        new_box.topOffset = box.topOffset / 2;
        new_box.center = box.center + (flipVectorOnQuad(box.topOffset, quadr) / 2);
        if (reachabilityEdgeInTheBox and (depth < MAX_DEPTH)) {
            recursive_kernel<<<1, 8>>>(new_box, input, leg, output, depth + 1);
        } else {
            // if (reachabilityEdgeInTheBox) {
            // distance = distance * 0;
            // }
            constexpr int blockSize = 1024 / 1;
            int numBlock = (input.length + blockSize - 1) / blockSize;
            fillOutKernel<<<numBlock, blockSize>>>(new_box, distance, input, output);
        }
    }
}
