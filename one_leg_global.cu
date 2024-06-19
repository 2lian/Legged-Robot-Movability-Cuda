// #include "HeaderCPP.h"
// #include "driver_types.h"
#include "one_leg.cu"

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

// constexpr Quaternion quat = {1, 0, 0, 0};
// constexpr Quaternion quat = {0.999, 0, 0.01, 0};
// constexpr Quaternion quat = {0.996, 0, -0.087, 0};
constexpr Quaternion quat = {0.985, 0, 0.174, 0};
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

template <typename Tin, typename Tindex>
__host__ __device__ __forceinline__ bool readBit(Tin data, Tindex bitPosition) {
    return (data >> bitPosition) & 1;
}

template <typename Tin>
__device__ __forceinline__ float3 flipVectorOnQuad(float3 vect, Tin quadNumber) {
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
        const auto in = input.elements[i];
        auto& out = output.elements[i];
        float3 delta = in - box.center;
        auto boxEdge = abs(box.topOffset);
        bool inside =
            boxEdge.x >= delta.x and boxEdge.y >= delta.y and boxEdge.z >= delta.z;
        inside = inside and -boxEdge.x < delta.x and -boxEdge.y < delta.y and
                 -boxEdge.z < delta.z;
        if (inside) {
            // out = make_float3(222, 222, 222);
            out = distance;
            // atomicAdd(&(output.elements[i].x), distance.x);
            // atomicAdd(&(output.elements[i].x), 1.f);
            // atomicAdd(&(output.elements[i].z), 1.f);
        }
    }
}

template <typename Tin>
__forceinline__ __device__ __host__ Tin bitShiftAboveN(Tin value, uint index,
                                                       uint shift) {
    // Create a mask to isolate bits above N
    Tin mask = ~((1 << (index + 0)) - 1);

    // Isolate the bits above N
    Tin bitsAboveN = value & mask;

    // Shift the isolated bits to the left by one
    Tin shiftedBits = bitsAboveN << shift;

    // Clear the original bits above N in the value
    value &= ~mask;

    // Combine the shifted bits with the original value
    value |= shiftedBits;

    return value;
}

template <typename Tin>
__forceinline__ __device__ __host__ Tin bitShiftBetween(Tin value, uint indexLow,
                                                        uint indexUp, uint shift) {
    // Create a mask to isolate bits between indexLow and indexUp
    Tin mask = ((1 << (indexUp + 1)) - 1) ^ ((1 << indexLow) - 1);

    // Isolate the bits within the range
    Tin bitsInRange = value & mask;

    // Shift the isolated bits to the left by the specified number of positions
    Tin shiftedBits = (bitsInRange << shift) & mask;

    // Clear the original bits in the range in the value
    value &= ~mask;

    // Combine the shifted bits with the original value
    value |= shiftedBits;

    return value;
}

template <typename T>
__host__ __forceinline__ __device__ T swapBits(T value, uint pos1, uint pos2) {
    // Extract the bits at pos1 and pos2
    T bit1 = (value >> pos1) & 1;
    T bit2 = (value >> pos2) & 1;

    // If the bits are different, swap them
    if (bit1 != bit2) {
        // Toggle the bits at pos1 and pos2
        value ^= (1 << pos1);
        value ^= (1 << pos2);
    }

    return value;
}
__forceinline__ __device__ __host__ uint reverse(uint bits) {
    bits = (((bits & 0xaaaaaaaa) >> 1) | ((bits & 0x55555555) << 1));
    bits = (((bits & 0xcccccccc) >> 2) | ((bits & 0x33333333) << 2));
    bits = (((bits & 0xf0f0f0f0) >> 4) | ((bits & 0x0f0f0f0f) << 4));
    bits = (((bits & 0xff00ff00) >> 8) | ((bits & 0x00ff00ff) << 8));
    return ((bits >> 16) | (bits << 16));
}

// constexpr cudaStream_t streams[10];
#define MAX_DEPTH 20
constexpr float BOXSIZE = 0.1;
constexpr uint SUB_QUAD = 2;

constexpr float MIN_BOX_X = BOXSIZE;
constexpr float MIN_BOX_Y = BOXSIZE;
constexpr float MIN_BOX_Z = BOXSIZE;
__global__ void recursive_kernel(Box box, const Array<float3> input,
                                 const LegDimensions leg, Array<float3> output,
                                 uchar depth, float rad, bool validity) {

    uint index = blockIdx.x * blockDim.x + threadIdx.x;
    uint stride = blockDim.x * gridDim.x;
    uint quadCount = 3;
    bool small[3];
    small[0] = abs(box.topOffset.x) < MIN_BOX_X;
    small[1] = abs(box.topOffset.y) < MIN_BOX_Y;
    small[2] = abs(box.topOffset.z) < MIN_BOX_Z;
    if (small[0]) { // too small from the start
        quadCount -= 1;
    }
    if (small[1]) { // too small from the start
        quadCount -= 1;
    }
    if (small[2]) { // too small from the start
        quadCount -= 1;
    }
    // quad_count = 3;
    const uint max_quad_ind = pow(pow(2, quadCount), SUB_QUAD);
    for (uint computeIndex = index; computeIndex < max_quad_ind; computeIndex += stride) {
        uint quadr = computeIndex;
        quadr = reverse(quadr) >> (32 - (quadCount * SUB_QUAD));

        Box new_box = box;
        float* offsets[3] = {&new_box.topOffset.x, &new_box.topOffset.y,
                             &new_box.topOffset.z};

        uint missingQuad;
        for (uint iter = 0; iter < SUB_QUAD; iter++) {
            uint sub_quadrant = quadr >> (iter * quadCount); // 111011000 -> 000111011
            sub_quadrant =                                   // 111011 -> 000000011
                sub_quadrant & ((1 << (quadCount)) - 1 + (1 << (quadCount)));
            float3 offsetDiv = make_float3(2, 2, 2);
            uint upperIndex = quadCount - 1;
            missingQuad = 0;
            float* divs[3] = {&offsetDiv.x, &offsetDiv.y, &offsetDiv.z};
            constexpr float minOffsets[3] = {MIN_BOX_X, MIN_BOX_Y, MIN_BOX_Z};
            for (uint q = 0; q < 3; q++) {
                float& off = *offsets[q];
                if (off < minOffsets[q]) {
                    missingQuad++;

                    if (readBit(sub_quadrant, upperIndex) && !small[q])
                        goto nextIndex; // do not executes on half of the quadr
                    sub_quadrant = bitShiftBetween(sub_quadrant, q, 3, 1);

                    if (small[q])
                        upperIndex++;
                    *divs[q] = 1;
                }
            }
            auto oldOff = new_box.topOffset;
            new_box.topOffset = new_box.topOffset / offsetDiv;
            auto offsetMovement = oldOff - new_box.topOffset;
            // offsetMovement = offsetMovement * -1;
            // if (iter & 1) {offsetMovement = offsetMovement * -1;}
            new_box.center =
                new_box.center + flipVectorOnQuad(offsetMovement, sub_quadrant);
        }
        {
            uint subQuadCount = 3 - missingQuad;
            bool tooSmall = subQuadCount <= 0;
            bool notCloseEnough;
            float3 centerToBox;
            if constexpr (SUB_QUAD <= 1) {
                notCloseEnough = false;
            } else {
                centerToBox =
                    max(abs(new_box.center - box.center) + abs(new_box.topOffset), 0);
                notCloseEnough = (linorm(centerToBox) < rad);
                centerToBox = make_float3(linorm(centerToBox) - rad, 0, 0);
            }

            // new_box.center = box.center + centerOffset;

            auto distance = new_box.center;
            // distance.y = 0.1;
            bool reachabilityEdgeInTheBox = not notCloseEnough;
            bool reachability = validity;
            if (reachabilityEdgeInTheBox) {
                reachability = distance_global(distance, leg, quat);
                reachabilityEdgeInTheBox = linorm(distance) < linorm(new_box.topOffset);
            } else
                distance = centerToBox;
            // bool tooSmall = (abs(new_box.topOffset.x) < MIN_BOX_X * 1) and
            // (abs(new_box.topOffset.y) < MIN_BOX_Y * 1) and
            // (abs(new_box.topOffset.z) < MIN_BOX_Z * 1);
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
            } else {
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
    nextIndex:;
    }
}
