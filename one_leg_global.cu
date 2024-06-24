#include "octree_util.cu.h"
#include "one_leg.cu"
#include "settings.h"
#include "thrust/detail/tuple.inl"
#include "thrust/tuple.h"
#include "unified_math_cuda.cu.h"
#include <iostream>
#include <ostream>

__device__ __forceinline__ thrust::tuple<float3, bool>
simpleGravityCheck(const float3 target, const Quaternion orientation, LegDimensions dim) {
    float3 coxaJoint =
        make_float3(cos(dim.body_angle) * dim.body, sin(dim.body_angle) * dim.body, 0);
    float3 rotatedCoxa = qtRotate(orientation, coxaJoint);
    float3 oriToCoxaHoriz = rotatedCoxa * make_float3(1, 1, 0);
    float3 coxaOrigTarget = target - coxaJoint;
    float dotprod = dot(coxaOrigTarget, oriToCoxaHoriz);
    bool validity = dotprod > 0;
    float3 planeToPoint = oriToCoxaHoriz * dotprod / linormRaw(oriToCoxaHoriz);
    // float3 projection = coxaOrigTarget - oriToCoxaHoriz *
    // dot(coxaOrigTarget, oriToCoxaHoriz) /
    // linormRaw(oriToCoxaHoriz);
    return thrust::make_tuple(planeToPoint, validity);
}

__host__ __device__ __forceinline__ void
z_rotateInPlace(float3& point, float z_rot, float& cos_memory, float& sin_memory) {
    sincosf(z_rot, &sin_memory, &cos_memory);
    float buffer = point.x * sin_memory;
    point.x = point.x * cos_memory - point.y * sin_memory;
    point.y = buffer + point.y * cos_memory;
}

__host__ __device__ __forceinline__ void
z_unrotateInPlace(float3& point, float& cos_memory, float& sin_memory) {
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
__forceinline__ __host__ __device__ LegDimensions rotate_leg_data(Quaternion quat,
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

__forceinline__ __host__ __device__ float3 make_asif_leg0(float3 point, LegDimensions leg,
                                                          float& cos_memory,
                                                          float& sin_memory) {
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
    // LegDimensions oriented_leg_dim = rotate_leg_data(quat, dim);
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
__host__ __device__ __forceinline__ Tout reachability_global(const float3& point,
                                                             const LegDimensions& dim,
                                                             const Quaternion quat) {
#ifdef __CUDA_ARCH__
    // LegDimensions oriented_leg_dim = rotate_leg_data(quat, dim);
    __shared__ LegDimensions oriented_leg_dim;
    if (threadIdx.x == 0) {
        oriented_leg_dim = rotate_leg_data(quat, dim);
    }
#else
    LegDimensions oriented_leg_dim = rotate_leg_data(quat, dim);
#endif

    auto unrotated_point = qtInvRotate(quat, point);
    float cos_memory;
    float sin_memory;

#ifdef __CUDA_ARCH__
    __syncthreads();
#endif
    auto point_as_leg0 =
        make_asif_leg0(unrotated_point, oriented_leg_dim, cos_memory, sin_memory);
    Tout result = reach_function(point_as_leg0, oriented_leg_dim);
    return result;
}

__host__ void reachability_kernel_cpu(const Array<float3> input, const LegDimensions dim,
                                      Array<bool> output) {
    for (int i = 0; i < input.length; i += 0) {
        std::cout << "hey" << std::endl;
        output.elements[i] = reachability_global(input.elements[i], dim, quatTest);
    }
}

__global__ void reachability_global_kernel(const Array<float3> input,
                                           const LegDimensions dim, Array<bool> output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        output.elements[i] = reachability_global(input.elements[i], dim, quatTest);
    }
}
__global__ void distance_global_kernel(const Array<float3> input, const LegDimensions dim,
                                       Array<float3> output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < input.length; i += stride) {
        float3 result = input.elements[i];
        bool reachability = distance_global(result, dim, quatTest);
        output.elements[i] = result;
    }
}

__global__ void recursive_kernel(Box box, const Array<float3> input,
                                 const LegDimensions leg, Array<float3> output,
                                 uchar depth, float rad, bool validity) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    uchar quadCount = 3;
    bool small[3];
    small[0] = abs(box.topOffset.x) < MIN_BOX_X;
    small[1] = abs(box.topOffset.y) < MIN_BOX_Y;
    small[2] = abs(box.topOffset.z) < MIN_BOX_Z;
    if (small[0])
        quadCount -= 1;
    if (small[1])
        quadCount -= 1;
    if (small[2])
        quadCount -= 1;

    // quad_count = 3;
    const uint max_quad_ind = pow(pow(2, quadCount), SUB_QUAD);
    for (uint computeIndex = index; computeIndex < max_quad_ind; computeIndex += stride) {

        Box new_box;
        uchar missingQuad;
        CreateChildBox(box, new_box, quadCount, computeIndex, small, missingQuad);

        if (missingQuad == DEADQUADRAN)
            continue;

        auto subQuadCount = 3 - missingQuad;
        bool tooSmall = subQuadCount <= 0;
        bool notCloseEnough;
        float3 distToNewBox;
        if constexpr (SUB_QUAD <= 1) {
            notCloseEnough = false;
        } else {
            distToNewBox =
                maxi(abs(new_box.center - box.center) + abs(new_box.topOffset), 0);
            notCloseEnough = (linorm(distToNewBox) < rad);
        }

        auto distance = new_box.center;
        // distance.y = 0.1;
        bool reachabilityEdgeInTheBox = not notCloseEnough;
        bool reachability = validity;
        if (reachabilityEdgeInTheBox) {
            reachability = distance_global(distance, leg, quatTest);
            reachabilityEdgeInTheBox = linorm(distance) < linorm(new_box.topOffset);
        } else
            distance = make_float3(linorm(distToNewBox) - rad, 0, 0);

        auto radius = linorm(distance);
        if (reachabilityEdgeInTheBox and (not tooSmall) and (depth < MAX_DEPTH)) {
            // auto subDistance =
            uint childCount = pow(pow(2, subQuadCount), SUB_QUAD);
            constexpr uint maxBlockSize = 1024 / 4;
            int blockSize = min(childCount, maxBlockSize);
            int numBlock = (childCount + blockSize - 1) / blockSize;
            // numBlock = 1;
            // blockSize = min(childCount, 24);
            // blockSize = 24;
            auto validity = reachability;
            recursive_kernel<<<numBlock, blockSize>>>(new_box, input, leg, output,
                                                      depth + 1, radius, validity);
        } else if constexpr (OutputOctree) {
            // distance = make_float3(min(radius, (float)500), 0, 0);
            distance = make_float3(depth, 0, 0);
            // distance = make_float3((threadIdx.x/24) + 0.01, 0, 0);
            if (tooSmall) {
                // distance = distance * 0;
            }
            if (depth == MAX_DEPTH) {
                // distance = make_float3(0,0,0);
            }
            if (reachability and not reachabilityEdgeInTheBox) {
                // distance = make_float3(-1,0,0);
            }

            constexpr int blockSize = 1024 / 4;
            int numBlock = (input.length + blockSize - 1) / blockSize;
            fillOutKernel<<<numBlock, blockSize>>>(new_box, distance, input, output);
        }
    }
}

__device__ bool distance(float3& point, const LegDimensions& dim, const Quaternion quat) {
    if constexpr (enableGrav) {
        const auto res = simpleGravityCheck(point, quat, dim);
        bool gravValid = thrust::get<1>(res);
        float3 gravDist = thrust::get<0>(res);
        if (not gravValid) {
            point = gravDist;
            return false;
        }
    }
    return distance_global(point, dim, quat);
}
