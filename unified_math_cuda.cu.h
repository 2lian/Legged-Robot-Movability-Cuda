#pragma once
// #include "cuda_runtime_api.h"
#include <driver_types.h>

typedef float4 Quaternion;

__host__ __device__ float3 qtRotate(Quaternion q, const float3 v);

__host__ __device__ Quaternion qtInvert(Quaternion quat);

__host__ __device__ float3 qtInvRotate(Quaternion q, const float3 v);

__host__ __device__ Quaternion qtMultiply(Quaternion q1, Quaternion q2);

__host__ __device__ Quaternion quatFromVectAngle(float3 axis, float angle);

struct QuaternionFunctor {
    Quaternion quat;

    __host__ __device__ QuaternionFunctor(Quaternion _quat) : quat(_quat) {}

    __device__ float3 operator()(float3 point) const {
        return qtRotate(quat, point);
        ;
    }
};

struct UnQuaternionFunctor {
    Quaternion quat;

    __host__ __device__ UnQuaternionFunctor(Quaternion _quat) : quat(_quat) {}

    __device__ float3 operator()(float3 point) const {
        return qtInvRotate(quat, point);
        ;
    }
};

__host__ __device__ float3 rpyFromQuat(const Quaternion q);
