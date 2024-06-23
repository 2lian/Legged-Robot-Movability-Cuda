#pragma once
#include "HeaderCUDA.h"
#include "cuda_device_runtime_api.h"
#include "cuda_runtime_api.h"
#include "driver_types.h"
#include "settings.h"
#include "unified_math_cuda.cu.h"
#include <iostream>
#include <ostream>

typedef struct Node {
    Box box;
    bool validity = false;
    bool leaf = false;
    bool raw = false;
    bool onEdge = false;
    uchar childrenCount = MaxChildQuad;
    Node* childrenArr;
} Node;

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

template <typename Tin, typename Tindex>
__host__ __device__ __forceinline__ bool readBit(Tin data, Tindex bitPosition) {
    return (data >> bitPosition) & 1;
}

template <typename Tin>
__device__ __host__ __forceinline__ float3 flipVectorOnQuad(float3 vect, Tin quadNumber) {
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

__forceinline__ __device__ __host__ uint CreateChildBox(Box parent, Box& child_out,
                                                        uchar quadCount,
                                                        uint computeIndex,
                                                        const bool* const dimensionSmall,
                                                        uchar& missingQuad_out) {
    child_out = parent;
    uint quadr = computeIndex;
    quadr = reverse(quadr) >> (32 - (quadCount * SUB_QUAD));
    float* offsets[3] = {&child_out.topOffset.x, &child_out.topOffset.y,
                         &child_out.topOffset.z};
    uint resultQuad = 0;
    for (uint iter = 0; iter < SUB_QUAD; iter++) {
        uint sub_quadrant = quadr >> (iter * quadCount); // 111011000 -> 000111011
        sub_quadrant =                                   // 111011 -> 000000011
            sub_quadrant & ((1 << (quadCount)) - 1 + (1 << (quadCount)));
        float3 offsetDiv = make_float3(2, 2, 2);
        uint upperIndex = quadCount - 1;
        missingQuad_out = 0;
        float* divs[3] = {&offsetDiv.x, &offsetDiv.y, &offsetDiv.z};
        constexpr float minOffsets[3] = {MIN_BOX_X, MIN_BOX_Y, MIN_BOX_Z};
        for (uint q = 0; q < 3; q++) {
            float& off = *offsets[q];
            if (off < minOffsets[q]) {
                missingQuad_out++;

                if (readBit(sub_quadrant, upperIndex) && !dimensionSmall[q]) {
                    missingQuad_out = DEADQUADRAN; // do not executes on half of the quadr
                    return resultQuad;
                }
                sub_quadrant = bitShiftBetween(sub_quadrant, q, 3, 1);

                if (dimensionSmall[q])
                    upperIndex++;
                *divs[q] = 1;
            }
        }
        auto oldOff = child_out.topOffset;
        child_out.topOffset = child_out.topOffset / offsetDiv;
        auto offsetMovement = oldOff - child_out.topOffset;
        // offsetMovement = offsetMovement * -1;
        // if (iter & 1) {offsetMovement = offsetMovement * -1;}
        child_out.center =
            child_out.center + flipVectorOnQuad(offsetMovement, sub_quadrant);
        resultQuad = resultQuad | (sub_quadrant << (iter * 3));
    }
    return resultQuad;
}

__forceinline__ __device__ __host__ bool isInBox(float3 vect, Box box) {
    auto boxEdge = abs(box.topOffset);
    bool inside = boxEdge.x >= vect.x and boxEdge.y >= vect.y and boxEdge.z >= vect.z;
    inside =
        inside and -boxEdge.x < vect.x and -boxEdge.y < vect.y and -boxEdge.z < vect.z;
    return inside;
}

__global__ void fillOutKernel(Box box, float3 distance, Array<float3> input,
                              Array<float3> output);

__forceinline__ __host__ __device__ Quaternion RPYtoQuat(float r, float p, float y) {
    Quaternion quatRoll = quatFromVectAngle(make_float3(1, 0, 0), r);
    // quatRoll = qtMultiply(quatRoll, quatInit);
    Quaternion quatPitch = quatFromVectAngle(make_float3(0, 1, 0), p);
    quatPitch = qtMultiply(quatPitch, quatRoll);
    Quaternion quatYaw = quatFromVectAngle(make_float3(0, 0, 1), y);
    quatYaw = qtMultiply(quatYaw, quatPitch);
    return quatYaw;
}

#define CUDA_CHECK_ERROR(errorMessage)                                                   \
    do {                                                                                 \
        cudaError_t err = cudaGetLastError();                                            \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error in %s: %s\n", errorMessage,                      \
                    cudaGetErrorString(err));                                            \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

__forceinline__ __host__ __device__ Quaternion QuaternionFromAngleIndex(uint AngleIndex) {
    auto reducedAngleIndex = AngleIndex;
    float rpy[3];
    for (uchar i = 0; i < 3; i++) {
        auto& maxInd = AngleSample_D[i];
        uchar ind = AngleIndex % maxInd;
        ind = (ind + (ind / 2)) % maxInd; // starts at middle
        reducedAngleIndex = AngleIndex / maxInd;

        float x = (float)ind / (uchar)(max(maxInd - 1, 1));
        rpy[i] = (1 - x) * AngleMinMax_D[i * 2] + x * AngleMinMax_D[i * 2 + 1];
    }
    Quaternion quat = RPYtoQuat(rpy[0], rpy[1], rpy[2]);
    return quat;
}

__host__ void copyTreeOnCpu(Node* gpuRoot, Node*& root);

__host__ uint countLeaf(Node&);

__host__ void deleteTree(Node* node);

__forceinline__ __device__ __host__ bool isNullBox(Box box) {
    return box.center.x == NullBox.center.x and box.center.y == NullBox.center.y and
           box.center.z == NullBox.center.z and box.topOffset.x == NullBox.topOffset.x and
           box.topOffset.y == NullBox.topOffset.y and
           box.topOffset.z == NullBox.topOffset.z;
}
__host__ Array<float3> extractValidAsArray(Node node);
